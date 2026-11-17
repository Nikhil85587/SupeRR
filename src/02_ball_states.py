"""Step 1: Rebuild the match state before every ball.
Output: data/processed/balls.parquet (one row per delivery, innings 1-2 + super overs)

Key data facts (see README.md):
- `runs` = bat runs only. In the UPDATED data set `extras1_runs` holds the actual runs of the extra (wide 1-5, no-ball
  1, bye / leg-bye 1-4, penalty 5), so extra runs = `extras1_runs` + `extras_2_runs`. (The original data set held an
  extras type code there and every extra had to be counted as 1 run.) 'other' behaves like a no-ball: not a legal ball.
- Targets are rebuilt and then made consistent with the official result text.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"

DATA = ROOT / "Updated Data"   # supplied CSVs (updated set); the original files in the project root are never touched
b = pd.read_csv(DATA / "ball_by_ball.csv", low_memory=False)
m = pd.read_csv(DATA / "matches.csv")
c = pd.read_csv(DATA / "competitions.csv")

# ---------- player ID fixes ----------
# Same person with two IDs (checked: both play for UAE). Abdul Samad and Haider Ali pairs are different people.
ID_MERGE = {39444: 8638}  # Basil Hameed
for col in ["striker_id", "non_striker_id", "bowler_id", "dismissed_player_id"]:
    b[col] = b[col].replace(ID_MERGE)

# ---------- ball-level basics ----------
b["extras_type"] = b["extras_type"].fillna("None")
b["legal"] = ~b["extras_type"].isin(["wide", "Overstep", "other"])
b["extra_runs"] = b["extras1_runs"] + b["extras_2_runs"]   # actual extra runs (penalty = 5 is already in extras1_runs)
b["total_runs"] = b["runs"] + b["extra_runs"]
# A batter "faces" every ball except wides; bowler is charged bat runs + wides + no-balls (not byes/leg-byes)
b["faced"] = b["extras_type"] != "wide"
b["bowler_runs"] = b["runs"] + np.where(b.extras_type.isin(["wide", "Overstep", "other"]), b["extras1_runs"] + b["extras_2_runs"], 0)
b["bowler_wicket"] = b["is_wicket"] & ~b["dismissal_type"].isin(["RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"])
b["batter_out"] = b["is_wicket"] & (b["dismissed_player_id"] == b["striker_id"])  # non-striker run-outs don't count against striker
b["is_super_over"] = b["innings"] > 2

b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
grp = b.groupby(["match_id", "innings"])
b["score_before"] = grp["total_runs"].cumsum() - b["total_runs"]
b["wkts_before"] = grp["is_wicket"].cumsum() - b["is_wicket"].astype(int)
b["legal_before"] = grp["legal"].cumsum() - b["legal"].astype(int)

# ---------- match context ----------
m = m.merge(c[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left")
m["competition"] = np.select(
    [m.comp_name.str.startswith("IPL"), m.match_format.eq("T20I")], ["IPL", "T20I"], "A-team")
m["margin_runs"] = m.result.str.extract(r"by (\d+) [Rr]un", expand=False).astype(float)
m["margin_wkts"] = m.result.str.extract(r"by (\d+) [Ww]icket", expand=False).astype(float)
m["tied"] = m.result.str.contains("Tied|tie", case=False, na=False)
m["no_result"] = m.result.str.contains("Abandon|No Result", case=False, na=False) | m.win_team_id.isna() | (m.win_team_id == 0) & ~m["tied"]

main = b[~b.is_super_over]
tot = main.groupby(["match_id", "innings"]).agg(total=("total_runs", "sum"), wk=("is_wicket", "sum"),
                                               bat_team=("batting_team_id", "first")).unstack()
tm = pd.DataFrame({
    "inn1_total": tot[("total", 1)], "inn2_total": tot[("total", 2)],
    "inn1_team": tot[("bat_team", 1)], "inn2_team": tot[("bat_team", 2)],
}).reset_index()
m = m.merge(tm, on="match_id", how="left")

# Target: start from rebuilt 1st-innings total + 1, then make it consistent with the official result.
t = m["inn1_total"] + 1
chase_won = m.win_team_id == m.inn2_team
defend_won = m.win_team_id == m.inn1_team
t = np.where(defend_won & m.margin_runs.notna(), m.inn2_total + m.margin_runs + 1, t)   # exact if chase total is right
t = np.where(chase_won, np.minimum(t, m.inn2_total), t)                                   # chase finished at/after target
t = np.where(m.tied, m.inn2_total + 1, t)
m["target"] = np.where(m.revised_target.notna(), m.revised_target, t)
m["target_source"] = np.where(m.revised_target.notna(), "revised (official)",
                     np.where(defend_won & m.margin_runs.notna(), "chase total + margin",
                     np.where(chase_won, "min(inn1+1, chase total)", "inn1 + 1")))
m["overs_inn1"] = m.first_inn_revised_overs.fillna(20)
m["overs_inn2"] = m.second_inn_revised_overs.fillna(m.overs_inn1)

b = b.merge(m[["match_id", "comp_id", "comp_name", "competition", "season", "match_date", "stage", "knock_out",
               "venue_id", "venue_country", "home_team_id", "away_team_id", "home_team", "away_team",
               "win_team_id", "no_result", "tied", "target", "overs_inn1", "overs_inn2"]], on="match_id", how="left")

b["overs_limit"] = np.where(b.innings == 1, b.overs_inn1, np.where(b.innings == 2, b.overs_inn2, 1))
b["balls_left"] = (b.overs_limit * 6 - b.legal_before).clip(lower=1)
b["wkts_left"] = (np.where(b.is_super_over, 2, 10) - b.wkts_before).clip(lower=1)
b["runs_needed"] = np.where(b.innings == 2, b.target - b.score_before, np.nan)
b["rrr"] = b.runs_needed / b.balls_left * 6
b["crr"] = np.where(b.legal_before > 0, b.score_before / b.legal_before.replace(0, np.nan) * 6, np.nan)
b["over_no"] = b["overid"]
b["phase"] = pd.cut(b.overid, [0, 6, 15, 20], labels=["Powerplay", "Middle", "Death"]).astype(str)
b["batting_team_won"] = np.where(b.no_result, np.nan, (b.win_team_id == b.batting_team_id).astype(float))
# Tied matches: decided by super over -> keep winner from win_team_id; if missing, 0.5
b.loc[b.tied & b.win_team_id.isna(), "batting_team_won"] = 0.5

b.to_parquet(P / "balls.parquet", index=False)
m.to_parquet(P / "matches_ext.parquet", index=False)

# ---------- quality report ----------
chk = m.dropna(subset=["inn1_total", "inn2_total"])
r = chk[chk.margin_runs.notna() & chk.revised_target.isna()]
err = r.inn1_total - r.inn2_total - r.margin_runs
print(f"balls: {len(b):,}  matches: {b.match_id.nunique()}")
print(f"1st-innings rebuild vs official run margin: exact {np.mean(err == 0):.0%}, within 3 runs {np.mean(err.abs() <= 3):.0%}, median abs {err.abs().median()}")
print("target source:", m.target_source.value_counts().to_dict())
print("matches with no result flag:", int(m.no_result.sum()))
print(b.groupby("competition").match_id.nunique().to_dict())
