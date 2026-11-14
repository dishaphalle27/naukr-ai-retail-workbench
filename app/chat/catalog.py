"""
CATALOG - the list of everything a question is allowed to touch.

  * dataset names, each with a VERSION label like  orders_retail@8a799afb3000
    (the label is a fingerprint of the file, so an answer can always be reproduced)
  * every column with a kind: numeric | date | text
  * hidden columns (personal data) that can never be selected, filtered or grouped
  * one ready-made joined view, "orders_wide", built with the SAFE join from Step 5

The "stage" says which version of the files is loaded: "_retail" (cleaned + normalized),
"_clean" (cleaned only). Version labels show it, so a raw-vs-cleaned mix-up is visible
in every answer.
"""
import hashlib
import re
from pathlib import Path

import pandas as pd

from app.cleaning.profiler import is_pii_column
from app.errors import DataLoadError
from app.retail.joins import join_orders_wide
from app.retail.metrics import num

MAX_TABLE_ROWS = 1_000_000          # memory guard
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
KIND_THRESHOLD = 0.9                # share of values that must fit for a column to count as numeric/date


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _column_kind(series):
    vals = series[series.notna() & (series.astype(str).str.strip() != "")].astype(str).str.strip()
    if len(vals) == 0:
        return "text"
    if vals.str.fullmatch(ISO_DATE).mean() >= KIND_THRESHOLD:
        return "date"
    if num(vals).notna().mean() >= KIND_THRESHOLD:
        return "numeric"
    return "text"


class Catalog:
    def __init__(self, tables, versions, join_reports=None, stage="_retail"):
        self.stage = stage
        self._raw = tables
        self._versions = versions
        self.join_reports = join_reports or {}
        self._cols = {}
        self._typed = {}
        self._nunique = {}
        for name, df in tables.items():
            if len(df) > MAX_TABLE_ROWS:
                raise DataLoadError("DATASET_TOO_LARGE", f"{name} has more than {MAX_TABLE_ROWS} rows")
            self._cols[name] = {c: {"kind": _column_kind(df[c]), "hidden": is_pii_column(c)} for c in df.columns}

    # ---------------------------------------------------------------- build
    @classmethod
    def from_folder(cls, folder, stage="_retail"):
        tables, versions = {}, {}
        for path in sorted(Path(folder).glob(f"*{stage}.csv")):
            base = path.stem[: -len(stage)]
            tables[base] = pd.read_csv(path, dtype=str, keep_default_na=False)
            versions[base] = f"{base}{stage}@{_sha(path)[:12]}"
        if not tables:
            raise DataLoadError("NO_DATASETS", f"No *{stage}.csv files found in {folder}")
        join_reports = {}
        # ready-made, pre-checked joined views: orders_wide and inventory_wide
        for base in ("orders", "inventory"):
            if base in tables and len(tables) > 1:
                wide, reports, _ = join_orders_wide(tables, base=base)
                if not reports:
                    continue
                name = f"{base}_wide"
                tables[name] = wide
                used = [base] + [r["right"] for r in reports]
                fingerprint = hashlib.sha256("|".join(versions[u] for u in used).encode()).hexdigest()[:12]
                versions[name] = f"{name}{stage}@{fingerprint}"
                join_reports[name] = reports
        return cls(tables, versions, join_reports, stage)

    # ---------------------------------------------------------------- lookups
    @property
    def datasets(self):
        return sorted(self._raw)

    def version(self, name):
        return self._versions[name]

    def columns(self, name):
        """All columns with kind and hidden flag."""
        return self._cols[name]

    def visible_columns(self, name):
        return {c: i for c, i in self._cols[name].items() if not i["hidden"]}

    def nunique(self, name, col):
        key = (name, col)
        if key not in self._nunique:
            s = self._raw[name][col]
            self._nunique[key] = int(s[s.notna() & (s.astype(str).str.strip() != "")].nunique())
        return self._nunique[key]

    def distinct_values(self, name, col, limit=200):
        """Distinct values of a column if there are few of them, else None."""
        if self.nunique(name, col) > limit:
            return None
        s = self._raw[name][col]
        s = s[s.notna() & (s.astype(str).str.strip() != "")]
        return sorted(str(v).strip() for v in s.unique())

    def table_text(self, name):
        """A text copy of a plain table, for the insight code. Refuses tables that hold personal data."""
        if any(i["hidden"] for i in self._cols[name].values()):
            raise DataLoadError("TABLE_HAS_PERSONAL_DATA", f"{name} is not available in raw form")
        return self._raw[name].copy()

    def row_count(self, name):
        return len(self._raw[name])

    def typed(self, name):
        """A typed COPY: numeric columns -> float, blanks -> NaN. Cached; callers must not modify it."""
        if name not in self._typed:
            df = self._raw[name].copy()
            for c, info in self._cols[name].items():
                s = df[c]
                if info["kind"] == "numeric":
                    df[c] = num(s).astype(float)
                else:
                    s = s.where(s.notna(), "").astype(str).str.strip()
                    df[c] = s.where(s != "", other=float("nan"))
            self._typed[name] = df
        return self._typed[name]

    def describe_for_prompt(self):
        """Schema only (no cell values). This is the only thing a model will ever be shown."""
        return {n: {c: i["kind"] for c, i in self.visible_columns(n).items()} for n in self.datasets}
