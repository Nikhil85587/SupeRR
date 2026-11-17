"""Step 2a: Extract men's T20I match results from the Cricsheet zip (info section only).
Output: data/processed/t20i_results.csv  (one row per match, all T20Is in Cricsheet)
"""
import json, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "data/external/cricsheet_t20i/t20s_male_json.zip"
OUT = ROOT / "data/processed/t20i_results.csv"

rows = []
with zipfile.ZipFile(ZIP) as z:
    for name in z.namelist():
        if not name.endswith(".json"):
            continue
        info = json.loads(z.read(name))["info"]
        out = info.get("outcome", {})
        ev = info.get("event", {}) or {}
        teams = info.get("teams", [None, None])
        rows.append(dict(
            cs_id=name[:-5],
            date=info["dates"][0],
            team1=teams[0], team2=teams[1],
            winner=out.get("winner"),
            result=out.get("result"),          # 'tie' / 'no result' / 'draw' / None
            eliminator=out.get("eliminator"),  # super-over winner when tied
            method=out.get("method"),
            venue=info.get("venue"), city=info.get("city"),
            event=ev.get("name"), match_number=ev.get("match_number"), stage=ev.get("stage"),
            match_type=info.get("match_type"),
        ))

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
# Tie decided by super over: ICC rankings treat as a tie; keep 'eliminator' separately.
df = df.sort_values(["date", "cs_id"]).reset_index(drop=True)
df.to_csv(OUT, index=False)
print(len(df), "matches", df.date.min().date(), "->", df.date.max().date())
print(df.result.value_counts(dropna=False).to_dict())
print("teams:", df[["team1", "team2"]].stack().nunique())
