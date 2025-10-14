
r"""
colddial.py

- Default paths (Windows):
    SOURCE_ROOT = r""
    DEST_ROOT   = r""

Usage:
    python colddial.py <filename.csv|filename.zip> [--source-root "..."] [--dest-root "..."]
                       [--no-overwrite] [--dnc-policy include|fallback|skip]

Features:
- Reads CSV or ZIP (auto-selects CSV inside).
- Embedded schema; no template required.
- Enforces destination column order from your template:
  First Name, Last Name, Address, City, State, Zip Code, Phone #1, Phone #2, Phone #3, Email
- Hard override mapping: Property* → Address/City/State/Zip Code; fallback to Mailing* if Property* missing.
- Owner name = FirstName + LastName (case/underscore-insensitive). Used for name-similarity scoring.
- Phone selection priority (max 3, deduped):
  1) Mobile (C1 before C2 if tie)
  2) Landline (C1 before C2)
  3) VOIP (lowest)
- **DNC policy (default = include):**
    include  → treat DNC same as others (Litigator still skipped)
    fallback → prefer non-DNC, but top-up with DNC if fewer than 3
    skip     → skip all DNC (may leave blanks)
- Prints sample name-similarity scores for first 5 rows.
"""
import argparse
import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
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
    "Phone #2",
    "Phone #3",
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

    if dest not in ["Phone #1", "Phone #2", "Phone #3"]:
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
    phone_cols = []
    type_for_phone: Dict[str, Optional[str]] = {}
    dnc_for_phone: Dict[str, Optional[str]] = {}
    lit_for_phone: Dict[str, Optional[str]] = {}
    contact_rank: Dict[str, int] = {}

    can = _canon
    for c in cols:
        n = can(c)
        if "phone" in n or "mobile" in n or "cell" in n or "tel" in n:
            phone_cols.append(c)

    canon_map = {can(c): c for c in cols}

    def pair_column(base_col: str, suffixes):
        bcan = can(base_col)
        for suf in suffixes:
            guesses = [
                f"{base_col}{suf}", f"{base_col} {suf}",
                base_col.replace(" ", "_") + f"_{suf.lstrip('_')}",
                base_col.replace("Phone", "Phone_") + suf,
            ]
            for g in guesses:
                cc = can(g)
                if cc in canon_map:
                    return canon_map[cc]
        for suf in suffixes:
            cc = bcan + can(suf)
            if cc in canon_map:
                return canon_map[cc]
        return None

    for c in phone_cols:
        type_for_phone[c] = pair_column(c, ["_Type", "Type"])
        dnc_for_phone[c] = pair_column(c, ["_DNC", "DNC"])
        lit_for_phone[c] = pair_column(c, ["_Litigator", "Litigator"])

    for c in phone_cols:
        n = can(c)
        if "contact1" in n or "owner1" in n or re.search(r'\bc1\b', c.lower() or ""):
            contact_rank[c] = 1
        elif "contact2" in n or "owner2" in n or re.search(r'\bc2\b', c.lower() or ""):
            contact_rank[c] = 2
        else:
            contact_rank[c] = 3

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

def best_phone_triplet(row: pd.Series,
                       dnc_policy: str,
                       phone_cols: List[str],
                       type_for_phone: Dict[str, Optional[str]],
                       contact_rank: Dict[str, int],
                       owner_full: str,
                       contact1_names: List[str],
                       contact2_names: List[str],
                       dnc_for_phone: Dict[str, Optional[str]],
                       lit_for_phone: Dict[str, Optional[str]]) -> Tuple[str, str, str]:
    candidates = []
    dnc_candidates = []
    seen = set()

    def cat(row, cols: List[str]) -> str:
        vals = [str(row[c]) for c in cols if c in row.index and pd.notna(row[c]) and str(row[c]).strip()]
        return " ".join(vals).strip()

    c1_full = cat(row, contact1_names)
    c2_full = cat(row, contact2_names)

    def name_sim(a: str, b: str) -> float:
        if not a.strip() or not b.strip():
            return 0.0
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    for col in phone_cols:
        raw = row.get(col, None)
        if pd.isna(raw) or not str(raw).strip():
            continue
        raw = str(raw).strip()
        if not is_phone_like(raw):
            continue
        phone = normalize_e164ish(raw)
        if not phone or phone in seen:
            continue

        # DNC / Litigator
        dnc_col = dnc_for_phone.get(col)
        lit_col = lit_for_phone.get(col)
        is_dnc = bool(dnc_col and dnc_col in row.index and is_truthy_flag(row.get(dnc_col)))
        is_lit = bool(lit_col and lit_col in row.index and is_truthy_flag(row.get(lit_col)))
        if is_lit:
            continue  # always skip litigator

        # Kind
        type_col = type_for_phone.get(col)
        type_val = row.get(type_col) if type_col else None
        kind = phone_kind_from_typeval(type_val)
        if kind == "unknown" and is_mobile_colname(col):
            kind = "mobile"  # column name hint

        # Contact rank
        c_rank = contact_rank.get(col, 3)

        # Name priority
        priority = 0.0
        if c_rank == 1:
            priority = max(priority, name_sim(owner_full, c1_full))
        elif c_rank == 2:
            priority = max(priority, name_sim(owner_full, c2_full))
        else:
            priority = max(name_sim(owner_full, c1_full), name_sim(owner_full, c2_full))

        # Scoring
        base = 0.0
        if kind == "mobile":
            base += 100
        elif kind == "land":
            base += 20
        elif kind == "voip":
            base += 10
        else:
            base += 15

        if c_rank == 1:
            base += 50
        elif c_rank == 2:
            base += 25

        base += 20.0 * priority

        target = candidates
        if is_dnc and dnc_policy in ('skip','fallback'):
            target = dnc_candidates if dnc_policy == 'fallback' else None
        if is_dnc and dnc_policy == 'skip':
            target = None
        if target is not None:
            target.append((base, phone, col))
            seen.add(phone)

    candidates.sort(key=lambda x: x[0], reverse=True)
    if dnc_policy == 'fallback' and len(candidates) < 3:
        dnc_candidates.sort(key=lambda x: x[0], reverse=True)
        for item in dnc_candidates:
            if all(item[1] != c[1] for c in candidates):
                candidates.append(item)
                if len(candidates) >= 3:
                    break

    phones = [c[1] for c in candidates[:3]]
    while len(phones) < 3:
        phones.append("")
    return tuple(phones[:3])

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
    dest_path = Path(args.dest_root) / f"{stem}_mapped.csv"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if args.no_overwrite and dest_path.exists():
        raise FileExistsError(f"Output exists and --no-overwrite specified: {dest_path}")

    src_df = read_source_dataframe(source_path)
    src_cols = list(src_df.columns)

    # Owner columns
    owner_first_src, owner_last_src = find_owner_name_columns(src_cols)

    # Build output
    out_df = pd.DataFrame(columns=DEST_HEADERS)

    # Map non-phone fields
    for dest in DEST_HEADERS:
        if dest in ["Phone #1", "Phone #2", "Phone #3"]:
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
    c1_name_cols = guess_contact_name_cols(src_cols, 1)
    c2_name_cols = guess_contact_name_cols(src_cols, 2)

    def _owner_full_from_row(r: pd.Series) -> str:
        f = str(r.get(owner_first_src, "")).strip() if owner_first_src else ""
        l = str(r.get(owner_last_src, "")).strip() if owner_last_src else ""
        return (f"{f} {l}").strip()

    owner_full_series = src_df.apply(_owner_full_from_row, axis=1)

    phones = src_df.apply(
        lambda row: best_phone_triplet(row, args.dnc_policy, phone_cols, type_for_phone, contact_rank,
                                       owner_full_series.loc[row.name], c1_name_cols, c2_name_cols,
                                       dnc_for_phone, lit_for_phone),
        axis=1, result_type='expand'
    )
    out_df["Phone #1"] = phones[0]
    out_df["Phone #2"] = phones[1]
    out_df["Phone #3"] = phones[2]

    # Enforce destination order and write
    out_df = out_df.reindex(columns=DEST_HEADERS, fill_value="")
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
