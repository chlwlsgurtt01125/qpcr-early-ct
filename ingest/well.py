# ingest/well.py
import re

WELL_RE = re.compile(r"^\s*([A-Za-z]{1,2})\s*0*([0-9]{1,2})\s*$")

def normalize_well(w: str) -> str | None:
    """
    Normalize well IDs:
      B3, b03, ' B 3 ' -> B03
      A12 -> A12
    Returns None if cannot parse.
    """
    if w is None:
        return None
    s = str(w).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    m = WELL_RE.match(s)
    if not m:
        return None
    row = m.group(1).upper()
    col = int(m.group(2))
    if col <= 0:
        return None
    return f"{row}{col:02d}"
