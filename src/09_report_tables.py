"""Step 6 helper: numbers for the methodology report (default ranking: min 150 pressure balls, both-teams pitch reading).
Outputs: outputs/report_*.csv and outputs/report_facts.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data/processed", ROOT / "outputs"
TH = 150

v = pd.read_parquet(P / "ball_values.parquet")
v["yr"] = v.season.astype(str).str[:4]
bat = pd.read_csv(OUT / f"step5_bat_min{TH}.csv")
bowl = pd.read_csv(OUT / f"step5_bowl_min{TH}.csv")
facts = []

# ---------- top-10 tables ----------
ids_bat = v[v.faced].groupby("striker_name").striker_id.first()
ids_bowl = v.groupby("bowler_name").bowler_id.first()


def seasons_positive(role, name):
    """Seasons (with >= 30 pressure balls) in which the player beat expectation under pressure."""
    if role == "bat":
        d = v[v.faced & v.bat_high & (v.striker_id == ids_bat[name])]
        g = d.groupby("yr").agg(n=("bat_value", "size"), val=("bat_value", "sum"))
    else:
        d = v[v.bowl_high & (v.bowler_id == ids_bowl[name])]
        g = d.groupby("yr").agg(n=("legal", "sum"), val=("bowl_value", "sum"))
    g = g[g.n >= 30]
    return f"{int((g.val > 0).sum())}/{len(g)}"


tb = bat.head(10).copy()
tb["seasons_positive"] = [seasons_positive("bat", n) for n in tb.player]
tb["team"] = tb.country.fillna("—")
tb.to_csv(OUT / "report_top10_bat.csv", index=False)
tw = bowl.head(10).copy()
tw["seasons_positive"] = [seasons_positive("bowl", n) for n in tw.player]
tw["team"] = tw.country.fillna("—")
tw.to_csv(OUT / "report_top10_bowl.csv", index=False)

for lab, t in [("BAT", tb), ("BOWL", tw)]:
    cols = (["rank", "player", "country", "ipl_teams", "pres_balls", "pressure_innings", "pres_SR", "pres_xSR", "pres_outs",
             "pres_x_outs"] if lab == "BAT" else
            ["rank", "player", "country", "ipl_teams", "pres_balls", "pressure_innings", "pres_econ", "pres_x_econ", "pres_wkts",
             "pres_x_wkts"]) + ["normal_level", "pressure_level", "rise", "consistency_pct", "bad_day_pct", "rank_5pct",
                                "rank_95pct", "pct_boot_top10", "rank_top15pct_cut", "rank_top25pct_cut", "rank_IPL_only",
                                "rank_T20I_only", "seasons_positive"]
    facts.append(f"\n{lab} top 10 (min {TH}):\n" + t[cols].round(1).to_string(index=False))

# ---------- pressure balls by competition for top 10 ----------
for lab, t, idmap, flt, high, n in [("BAT", tb, ids_bat, v.faced, "bat_high", None), ("BOWL", tw, ids_bowl, v.legal, "bowl_high", "legal")]:
    rows = []
    for nm in t.player:
        pid = idmap[nm]
        d = v[flt & v[high] & ((v.striker_id if lab == "BAT" else v.bowler_id) == pid)]
        rows.append((nm, int((d.competition == "IPL").sum()), int((d.competition == "T20I").sum()), int((d.competition == "A-team").sum())))
    facts.append(f"{lab} pressure balls IPL / T20I / A-team: " + "; ".join(f"{a} {b}/{c}/{d}" for a, b, c, d in rows))

# ---------- dataset + pool facts ----------
facts.append(f"\nBalls {len(v):,}; matches {v.match_id.nunique()}; IPL matches {v[v.competition=='IPL'].match_id.nunique()}; "
             f"T20I {v[v.competition=='T20I'].match_id.nunique()}; A-team {v[v.competition=='A-team'].match_id.nunique()}")
facts.append(f"Eligible pool at min {TH}: batters {len(bat)}, bowlers {len(bowl)}")
facts.append(f"Batters/bowlers who faced/bowled at least 1 ball: {v[v.faced].striker_id.nunique()} / {v.bowler_id.nunique()}")

# ---------- example match series for the chart ----------
b = pd.read_parquet(P / "balls_pressure.parquet")
ex = b[(b.home_team == "Rajasthan Royals") & (b.away_team == "Sunrisers Hyderabad") & (b.match_date.astype(str) == "2023-05-07")
       & (b.innings == 2)].sort_values("innings_ball_num")
ex = ex.assign(over_ball=ex.overid - 1 + ex.over_ball_num.clip(upper=6) / 6)
ser = ex[["innings_ball_num", "overid", "over_ball_num", "legal", "runs", "total_runs", "is_wicket", "runs_needed", "balls_left",
          "wp_bat", "LI", "pct_bat", "striker_name", "bowler_name"]]
ser.to_csv(OUT / "report_example_chase.csv", index=False)
facts.append(f"\nExample chase rows: {len(ser)}; target {int(ex.target.iloc[0])}; start WP {ex.wp_bat.iloc[0]:.2f}; "
             f"min WP {ex.wp_bat.min():.3f} at ball {int(ex.loc[ex.wp_bat.idxmin(), 'innings_ball_num'])}; max LI {ex.LI.max():.1f}")
last = ex.tail(8)[["overid", "over_ball_num", "runs", "total_runs", "is_wicket", "runs_needed", "balls_left", "wp_bat", "LI",
                   "striker_name", "bowler_name"]]
facts.append("Last 8 balls of the chase:\n" + last.round(2).to_string(index=False))

# ---------- importance facts ----------
imp = pd.read_parquet(P / "match_importance.parquet")
facts.append("\nImportance by competition: " + imp.groupby("competition").importance.agg(["mean", "max"]).round(2).to_string())
facts.append("Pressure-ball share by competition (batting): " +
             str(v[v.bat_high].competition.value_counts(normalize=True).round(3).to_dict()) +
             " | all balls: " + str(v.competition.value_counts(normalize=True).round(3).to_dict()))

txt = "\n".join(facts)
print(txt)
(OUT / "report_facts.txt").write_text(txt, encoding="utf-8")
