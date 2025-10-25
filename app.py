import os
from flask import Flask, request, send_file, render_template_string, abort, jsonify, make_response
import tempfile, sys, subprocess
from pathlib import Path
from io import BytesIO
from werkzeug.middleware.proxy_fix import ProxyFix
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BAN_THRESHOLD = int(os.getenv("BAN_THRESHOLD", "3"))
COUNT_TTL_SECONDS = int(os.getenv("COUNT_TTL_SECONDS", "0"))  # 0 = no TTL (true permanent counter)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN")  # set a random secret if you enable unban route

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
info = r.info("server")
print(REDIS_URL)
print("Server run_id:", r.info("server")["run_id"], "port:", r.info("server")["tcp_port"])
print("Selected DB (server view):", r.info("clients").get("tracking-clients", "n/a"))  # optional
print("APP Redis run_id:", info["run_id"], "port:", info["tcp_port"])

app = Flask(__name__)
# If you're behind a proxy/load balancer (nginx/haproxy), trust X-Forwarded-* safely:
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

BANNED_SET = "banned:ips"
COUNT_PREFIX = "ip404:"

def client_ip():
    # access_route respects ProxyFix and gives left-most (original) client IP
    return (request.access_route[0] if request.access_route else request.remote_addr) or "unknown"

@app.before_request
def block_banned_ips():
    ip = client_ip()
    if ip != "127.0.0.1" and r.sismember(BANNED_SET, ip):
        abort(403)  # Forbidden

@app.errorhandler(404)
def on_404(e):
    ip = client_ip()
    key = f"{COUNT_PREFIX}{ip}"

    # increment count
    count = r.incr(key)
    if COUNT_TTL_SECONDS > 0 and r.ttl(key) == -1:
        r.expire(key, COUNT_TTL_SECONDS)

    # if over threshold, ban permanently
    if count >= BAN_THRESHOLD:
        r.sadd(BANNED_SET, ip)

    # Return your normal 404 response (don’t leak ban info)
    return jsonify({"error": "Not Found"}), 404

@app.route('/favicon.ico')
@app.route('/apple-touch-icon-precomposed.png')
@app.route('/apple-touch-icon.png')
def favicon():
    return '', 204

@app.route("/robots.txt")
def robots_txt():
    body = "User-agent: *\nDisallow: /\n"
    resp = make_response(body)
    resp.headers["Content-Type"] = "text/plain; charset=utf-8"
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp

# --- Optional: admin helpers ---
@app.get("/_banlist")
def list_bans():
    token = request.headers.get("X-Admin-Token")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        abort(403)
    return jsonify(sorted(r.smembers(BANNED_SET)))

@app.post("/_unban/<ip>")
def unban(ip):
    token = request.headers.get("X-Admin-Token")
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        abort(403)
    r.srem(BANNED_SET, ip)
    r.delete(f"{COUNT_PREFIX}{ip}")
    return jsonify({"unbanned": ip})

UPLOAD_FORM = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>CRM and ColdDial Formatter</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 40px; color:#333; }
      .box { border: 2px dashed #aaa; padding: 24px; width: 520px; max-width: 90vw; text-align:center; }
      .row { margin: 12px 0; }
      label { display:inline-block; min-width:140px; text-align:right; margin-right: 8px; }
      select, input[type=file] { width: 260px; }
      button { padding: 8px 16px; margin-top: 10px; }
    </style>
  </head>
  <body>
    <h2>Upload XLeads file to convert to CRM or ColdDial</h2>
    <form method="post" enctype="multipart/form-data">
      <div class="box">
        <div class="row">
          <label for="file">File</label>
          <input id="file" type="file" name="file" accept=".zip,.csv" required>
        </div>
        <div class="row">
          <label for="dnc">DNC Policy</label>
          <select id="dnc" name="dnc_policy">
            <option value="include" selected>Include (default)</option>
            <option value="fallback">Fallback (prefer non-DNC, fill if needed)</option>
            <option value="skip">Skip (may leave blanks)</option>
          </select>
        </div>
        <div>
        <button type="submit" name="submit" value="xleads">Xleads</button>&nbsp;<button type="submit" name="submit" value="colddial">Colddial</button>
        </div>
      </div>
    </form>
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(UPLOAD_FORM)

    uploaded = request.files.get("file")
    if not uploaded:
        return "No file uploaded", 400

    ext = Path(uploaded.filename).suffix.lower()
    if ext not in {".zip", ".csv"}:
        return "Please upload a .zip or .csv file.", 400

    dnc_policy = request.form.get("dnc_policy", "include")
    if dnc_policy not in {"include", "fallback", "skip"}:
        dnc_policy = "include"
    
    target = request.form.get("submit", "colddial")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        src_path = tmpdir / uploaded.filename
        uploaded.save(src_path)

        if target == "xleads":
            out_path = tmpdir / f"{src_path.stem}_xleads.csv"
        else:
            out_path = tmpdir / f"{src_path.stem}_colddial.csv"

        try:
            # Run colddial.py with the chosen DNC policy
            if  target == "xleads":
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).parent / "xleads_ghl.py"),
                        src_path.name,
                        "--source-root", str(tmpdir),
                        "--dest-root", str(tmpdir),
                        "--dnc-policy", dnc_policy
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
                download_name = f"{src_path.stem}_xleads.csv"
            else:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).parent / "colddial.py"),
                        src_path.name,
                        "--source-root", str(tmpdir),
                        "--dest-root", str(tmpdir),
                        "--dnc-policy", dnc_policy
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )
                download_name = f"{src_path.stem}_colddial.csv"
        except subprocess.CalledProcessError as e:
            return f"<pre>Processing failed:\n\n{e.stderr or e.stdout}</pre>", 500

        if not out_path.exists():
            return "No CSV produced.", 500

        # Stream from memory to avoid Windows file-lock issues
        data = out_path.read_bytes()
        mem = BytesIO(data)
        mem.seek(0)
        
        return send_file(
            mem,
            as_attachment=True,
            download_name=download_name,
            mimetype="text/csv"
        )

if __name__ == "__main__":
    # Set debug=False in production
    app.run(host="0.0.0.0", port=5000, debug=False)
