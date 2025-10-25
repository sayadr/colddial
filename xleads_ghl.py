r"""
colddial.py (patched)

Changes:
- Adds "Phone #1 Type", "Phone #2 Type", "Phone #3 Type" to DEST_HEADERS.
- Reads phone type from the spreadsheet (the *_Type columns auto-detected in guess_phone_columns).
- Uses type for ranking (mobile > landline > voip > unknown), as before.
- Returns both numbers and their raw types from best_phone_triplet and writes them to output.
"""
import argparse
import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
import ghl_flat_mapper
import pandas as pd
from pathlib import Path
import zipfile

# ======================== CONFIG ========================
SOURCE_ROOT_DEFAULT = r""
DEST_ROOT_DEFAULT   = r""

DEST_HEADERS = [
    "First Name",
    "Last Name",
    "Address",
    "City",
    "State",
    "Zip Code",
    "Phone #1",
    "Phone #1 Type",   # <-- NEW
    "Phone #2",
    "Phone #2 Type",   # <-- NEW
    "Phone #3",
    "Phone #3 Type",   # <-- NEW
    "Email",
]

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "First Name": [
        "FirstName", "first_name", "firstname", "FIRSTNAME", "First Name",
        "owner 1 first name", "owner first name", "owner1firstname", "fname",
        "contact1 first name", "c1 first name", "primary first name"
    ],
    "Last Name": [
        "LastName", "last_name", "lastname", "LASTNAME", "Last Name",
        "owner 1 last name", "owner last name", "owner1lastname", "lname",
        "contact1 last name", "c1 last name", "primary last name"
    ],
    "Address": [
        "mailing address", "address", "address line 1", "address1", "street address",
        "situs address", "property address"
    ],
    "City": ["city", "mailing city", "property city", "situs city"],
    "State": ["state", "mailing state", "property state", "situs state"],
    "Zip Code": ["zip", "zip code", "postal code", "zipcode", "mailing zip", "mailing zip code"],
    "Email": ["email", "email address", "e-mail"],
    "Phone #1": [],
    "Phone #2": [],
    "Phone #3": [],
}

PHONE_COL_MAX = 50

# ======================== HELPERS ========================
def _canon(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def best_source_for(dest: str, src_cols: List[str]) -> Optional[str]:
    candidates = COLUMN_SYNONYMS.get(dest, [])
    src_lower_map = {c.lower(): c for c in src_cols}

    for syn in candidates:
        syn_l = syn.lower()
        if syn_l in src_lower_map:
            return src_lower_map[syn_l]
        for s in src_cols:
            if syn_l in s.lower():
                return s

    if dest not in ["Phone #1", "Phone #2", "Phone #3", "Phone #1 Type", "Phone #2 Type", "Phone #3 Type"]:
        best, best_ratio = None, 0.0
        for s in src_cols:
            r = SequenceMatcher(None, _canon(dest), _canon(s)).ratio()
            if r > best_ratio:
                best_ratio, best = r, s
        if best_ratio > 0.92:
            return best
    return None

def is_phone_like(val: str) -> bool:
    if not isinstance(val, str):
        val = str(val) if pd.notna(val) else ""
    digits = re.sub(r'\D', '', val)
    return len(digits) >= 10

def normalize_e164ish(val: str) -> str:
    if not isinstance(val, str):
        val = str(val) if pd.notna(val) else ""
    digits = re.sub(r'\D', '', val)
    if not digits:
        return ""
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    if val.startswith("+") and len(digits) >= 10:
        return "+" + digits
    return digits

def phone_kind_from_typeval(val: str) -> str:
    if not isinstance(val, str):
        val = str(val) if pd.notna(val) else ""
    v = val.lower()
    if any(k in v for k in ["mobile", "cell", "wireless"]):
        return "mobile"
    if "voip" in v:
        return "voip"
    if any(k in v for k in ["land", "landline", "home", "work", "office", "pstn", "residential", "business"]):
        return "land"
    return "unknown"

def is_truthy_flag(val) -> bool:
    s = (str(val) if val is not None else "").strip().lower()
    return s in {"1","true","yes","y","t"}

def is_mobile_colname(col: str) -> bool:
    n = _canon(col)
    return ("mobile" in n) or ("cell" in n)

def guess_contact_name_cols(cols: List[str], which: int) -> List[str]:
    labels = [f"contact {which}", f"contact{which}", f"c{which}"]
    candidates = []
    for c in cols:
        n = _canon(c)
        if any(lbl.replace(" ", "") in n for lbl in labels) and ("name" in n or "fullname" in n):
            candidates.append(c)
    return candidates


def guess_phone_columns(cols: List[str]) -> Tuple[List[str], Dict[str, Optional[str]], Dict[str, int], Dict[str, Optional[str]], Dict[str, Optional[str]]]:
    """
    Detect phone columns strictly matching the schema: Contact{N}Phone_{K}
    and pair *_Type, *_DNC, *_Litigator columns (case-insensitive).
    Returns:
      phone_cols: list of phone number column names
      type_for_phone: map phone_col -> its paired type column name (or None)
      contact_rank: map phone_col -> 1 for Contact1, 2 for Contact2, else 3
      dnc_for_phone: map phone_col -> its paired DNC column name (or None)
      lit_for_phone: map phone_col -> its paired Litigator column name (or None)
    """
    phone_cols: List[str] = []
    type_for_phone: Dict[str, Optional[str]] = {}
    dnc_for_phone: Dict[str, Optional[str]] = {}
    lit_for_phone: Dict[str, Optional[str]] = {}
    contact_rank: Dict[str, int] = {}

    # Build a case-insensitive lookup for exact column names
    lower_to_actual = {c.lower(): c for c in cols}

    # Regex for Contact{number}Phone_{index}
    pat = re.compile(r'^contact(\d+)phone_(\d+)$', re.IGNORECASE)

    for c in cols:
        m = pat.match(c)
        if not m:
            continue
        phone_cols.append(c)

        contact_num = int(m.group(1))
        contact_rank[c] = 1 if contact_num == 1 else (2 if contact_num == 2 else 3)

        base = c  # exact case
        # Pair columns: *_Type, *_DNC, *_Litigator (same base with suffix)
        for suffix, target_map in [('_type', type_for_phone), ('_dnc', dnc_for_phone), ('_litigator', lit_for_phone)]:
            pair_key = (base + suffix).lower()
            target_map[c] = lower_to_actual.get(pair_key)

    # Enforce PHONE_COL_MAX and stable order
    phone_cols = list(dict.fromkeys(phone_cols))[:PHONE_COL_MAX]
    return phone_cols, type_for_phone, contact_rank, dnc_for_phone, lit_for_phone

def _select_csv_member(zf: zipfile.ZipFile, preferred_stem: str = "") -> str:
    csv_members = [zi for zi in zf.infolist() if zi.filename.lower().endswith(".csv")]
    if not csv_members:
        raise FileNotFoundError("No CSV files found inside the ZIP.")
    if preferred_stem:
        def clean(s): return _canon(Path(s).stem)
        pref = clean(preferred_stem)
        for zi in csv_members:
            if clean(zi.filename) == pref:
                return zi.filename
    if len(csv_members) == 1:
        return csv_members[0].filename
    csv_members.sort(key=lambda z: z.file_size, reverse=True)
    return csv_members[0].filename

def read_source_dataframe(source_path: Path) -> pd.DataFrame:
    if source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path, "r") as zf:
            member_name = _select_csv_member(zf, source_path.stem)
            with zf.open(member_name, "r") as fh:
                return pd.read_csv(fh, dtype=str, keep_default_na=False)
    else:
        return pd.read_csv(source_path, dtype=str, keep_default_na=False)

def find_owner_name_columns(src_cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    first_col = None
    last_col = None
    for c in src_cols:
        cn = _canon(c)
        if cn in ("firstname", "ownerfirstname", "owner1firstname"):
            first_col = c if first_col is None else first_col
        if cn in ("lastname", "ownerlastname", "owner1lastname"):
            last_col = c if last_col is None else last_col
    if first_col is None:
        for c in src_cols:
            cn = _canon(c)
            if "first" in cn and "name" in cn:
                first_col = c; break
    if last_col is None:
        for c in src_cols:
            cn = _canon(c)
            if "last" in cn and "name" in cn:
                last_col = c; break
    return first_col, last_col

TYPE_SCORE = {
    "mobile": 100, "cell": 100, "wireless": 100,
    "residential": 70, "landline": 65,
    "business": 60,
    "voip": 55,
    "unknown": 50,
}

def _is_trueish(v: str) -> bool:
    return str(v).strip().lower() in ("true", "t", "1", "yes", "y")

def best_phone_triplet(
    row,
    dnc_policy: str,                     # "include" (default), "skip", or "fallback"
    phone_cols: List[str],
    type_for_phone: Dict[str, Optional[str]],  # e.g. {"Contact1_Phone_1":"Contact1_Phone_1_Type", ...}
    contact_rank: Dict[str, int],        # e.g. Contact1=1, Contact2=2, etc. per phone col
    owner_full: str,
    contact1_name_cols: List[str],
    contact2_name_cols: List[str],
    dnc_for_phone: Dict[str, Optional[str]],   # per-phone DNC flag column (optional)
    lit_for_phone: Dict[str, Optional[str]],   # per-phone litigation flag column (optional)
) -> Tuple[str, str, str, str, str, str]:
    """
    Returns six values in GROUPED order expected by your code:
      [0]=Phone #1, [1]=Phone #2, [2]=Phone #3,
      [3]=Phone #1 Type, [4]=Phone #2 Type, [5]=Phone #3 Type
    """

    def cat(cols: List[str]) -> str:
        parts = []
        for c in cols:
            if c in row:
                s = str(row[c]).strip()
                if s:
                    parts.append(s)
        return " ".join(parts)

    owner_l = (owner_full or "").strip().lower()
    c1_l = cat(contact1_name_cols).lower()
    c2_l = cat(contact2_name_cols).lower()

    candidates = []  # (score, phone_pos, contact_pos, phone, raw_type, phone_col)
    seen = set()

    for col in phone_cols:
        raw = str(row.get(col, "")).strip()
        phone = normalize_e164ish(raw)  # your helper
        if not is_phone_like(phone):    # your helper
            continue
        if phone in seen:
            continue

        # sibling/type/dnc/lit columns (optional)
        tcol = type_for_phone.get(col)
        raw_type = str(row.get(tcol, "")).strip() if tcol else ""
        kind = (raw_type or "").lower() or "unknown"

        dcol = dnc_for_phone.get(col)
        is_dnc = _is_trueish(row.get(dcol, "")) if dcol else False

        lcol = lit_for_phone.get(col)
        is_lit = _is_trueish(row.get(lcol, "")) if lcol else False

        # hard-exclude litigators
        if is_lit:
            continue
        # DNC policy
        if dnc_policy == "skip" and is_dnc:
            continue

        # score
        score = TYPE_SCORE.get(kind, TYPE_SCORE["unknown"])
        c_rank = contact_rank.get(col, 3)                # Contact1=1, Contact2=2, Contact3=3...
        score += {1: 6, 2: 3, 3: 0}.get(c_rank, 0)

        # light owner-name proximity nudge
        if owner_l and c1_l and (owner_l in c1_l or c1_l in owner_l):
            score += 2
        if owner_l and c2_l and (owner_l in c2_l or c2_l in owner_l):
            score += 1

        # fallback policy keeps DNC but penalizes
        if dnc_policy == "fallback" and is_dnc:
            score -= 8

        # tie-breaker #1: prefer *_phone_1 over *_phone_2 over *_phone_3
        m = re.search(r'phone[\s_]*(\d+)', col, re.I)
        phone_pos = int(m.group(1)) if m else 99

        candidates.append((score, phone_pos, c_rank, phone, raw_type, col))
        seen.add(phone)

    # sort: higher score first; then *_phone_1 before *_phone_2 before *_phone_3;
    # then Contact1 before Contact2 before Contact3; then phone string for stability
    candidates.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))

    top = candidates[:3]
    phones = [t[3] for t in top]
    types  = [t[4] for t in top]

    # pad to length 3 each
    while len(phones) < 3: phones.append("")
    while len(types)  < 3: types.append("")

    # IMPORTANT: grouped order to match your downstream indexing
    return phones[0], phones[1], phones[2], types[0], types[1], types[2]

# ======================== MAIN ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filename", help="Source CSV/ZIP filename (not full path)")
    ap.add_argument("--source-root", default=SOURCE_ROOT_DEFAULT, help="Override source root directory")
    ap.add_argument("--dest-root", default=DEST_ROOT_DEFAULT, help="Override destination root directory")
    ap.add_argument("--no-overwrite", action="store_true", help="Do not overwrite an existing output file")
    ap.add_argument("--dnc-policy", choices=["skip","fallback","include"], default="include",
                    help="How to treat numbers flagged DNC: skip, include only if needed (fallback), or always include (default)")
    args = ap.parse_args()

    source_path = Path(args.source_root) / args.filename
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    stem = Path(args.filename).stem
    dest_path = Path(args.dest_root) / f"{stem}_xleads.csv"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if args.no_overwrite and dest_path.exists():
        raise FileExistsError(f"Output exists and --no-overwrite specified: {dest_path}")

    src_df = read_source_dataframe(source_path)
    src_cols = list(src_df.columns)

    # Owner columns
    owner_first_src, owner_last_src = find_owner_name_columns(src_cols)

    # Build output
    HEADERS = src_cols + DEST_HEADERS
    out_df = src_df.copy()

    # Map non-phone fields
    for dest in DEST_HEADERS:
        if dest in ["Phone #1", "Phone #2", "Phone #3", "Phone #1 Type", "Phone #2 Type", "Phone #3 Type"]:
            continue
        src_match = best_source_for(dest, src_cols)
        out_df[dest] = src_df[src_match] if (src_match and src_match in src_df.columns) else ""

    # Force First/Last name mapping
    if owner_first_src and owner_first_src in src_df.columns:
        out_df["First Name"] = src_df[owner_first_src]
    if owner_last_src and owner_last_src in src_df.columns:
        out_df["Last Name"] = src_df[owner_last_src]

    # Force address mapping with Property* override and Mailing* fallback
    if "PropertyAddress" in src_df.columns:
        out_df["Address"] = src_df["PropertyAddress"]
    elif "MailingAddress" in src_df.columns:
        out_df["Address"] = src_df["MailingAddress"]

    if "PropertyCity" in src_df.columns:
        out_df["City"] = src_df["PropertyCity"]
    elif "MailingCity" in src_df.columns:
        out_df["City"] = src_df["MailingCity"]

    if "PropertyState" in src_df.columns:
        out_df["State"] = src_df["PropertyState"]
    elif "MailingState" in src_df.columns:
        out_df["State"] = src_df["MailingState"]

    if "PropertyPostalCode" in src_df.columns:
        out_df["Zip Code"] = src_df["PropertyPostalCode"]
    elif "MailingPostalCode" in src_df.columns:
        out_df["Zip Code"] = src_df["MailingPostalCode"]
    elif "MailingZip" in src_df.columns:
        out_df["Zip Code"] = src_df["MailingZip"]
    elif "Zip" in src_df.columns:
        out_df["Zip Code"] = src_df["Zip"]

    # Phones with DNC policy handling and VOIP deprioritized
    phone_cols, type_for_phone, contact_rank, dnc_for_phone, lit_for_phone = guess_phone_columns(src_cols)
    # Using explicit Contact{N}Phone_{K} schema; no further filtering needed.

    c1_name_cols = guess_contact_name_cols(src_cols, 1)
    c2_name_cols = guess_contact_name_cols(src_cols, 2)

    def _owner_full_from_row(r: pd.Series) -> str:
        f = str(r.get(owner_first_src, "")).strip() if owner_first_src else ""
        l = str(r.get(owner_last_src, "")).strip() if owner_last_src else ""
        return (f"{f} {l}").strip()

    owner_full_series = src_df.apply(_owner_full_from_row, axis=1)

    phones_types = src_df.apply(
        lambda row: best_phone_triplet(row, args.dnc_policy, phone_cols, type_for_phone, contact_rank,
                                       owner_full_series.loc[row.name], c1_name_cols, c2_name_cols,
                                       dnc_for_phone, lit_for_phone),
        axis=1, result_type='expand'
    )
    # phones_types columns: 0,1,2 = phones; 3,4,5 = raw types from sheet
    out_df["Phone #1"] = phones_types[0]
    out_df["Phone #1 Type"] = phones_types[3]
    out_df["Phone #2"] = phones_types[1]
    out_df["Phone #2 Type"] = phones_types[4]
    out_df["Phone #3"] = phones_types[2]
    out_df["Phone #3 Type"] = phones_types[5]

    # Enforce destination order and write
    out_df = out_df.reindex(columns=HEADERS, fill_value="")
    out_df = ghl_flat_mapper.map_source_to_destination(out_df)
    if "MarketValue" in out_df.columns:
        out_df = out_df.rename(columns={"MarketValue": "AVM"})
    if "AssessedTotal" in out_df.columns:
        out_df = out_df.drop(columns=["AssessedTotal"])
    out_df.to_csv(dest_path, index=False)

    # Debug prints
    print("Name similarity samples (first 5 rows):")
    for i in range(min(5, len(src_df))):
        owner = owner_full_series.iloc[i]
        c1n = " ".join([str(src_df.loc[i, c]) for c in c1_name_cols if c in src_df.columns and str(src_df.loc[i, c]).strip()])
        c2n = " ".join([str(src_df.loc[i, c]) for c in c2_name_cols if c in src_df.columns and str(src_df.loc[i, c]).strip()])
        sim1 = SequenceMatcher(None, owner.lower(), c1n.lower()).ratio() if owner and c1n else 0.0
        sim2 = SequenceMatcher(None, owner.lower(), c2n.lower()).ratio() if owner and c2n else 0.0
        print(f"Row {i}: owner='{owner}' | c1='{c1n}' (sim={sim1:.2f}) | c2='{c2n}' (sim={sim2:.2f})")

    print(f"Wrote {len(out_df)} rows to {dest_path}")
    print("Detected phone columns:", phone_cols)
    print("Owner first/last columns:", owner_first_src, owner_last_src)
    print("Contact1 name columns:", c1_name_cols)
    print("Contact2 name columns:", c2_name_cols)

if __name__ == "__main__":
    main()
