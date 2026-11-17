"""Collect step-5 rankings into one workbook for manual review: outputs/step5_rankings.xlsx"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
NOTES = [
    ("How to read", ""),
    ("rank", "Final rank = 62.5% pressure level + 22.5% consistency + 15% rise vs own normal level (z-scores within the eligible pool)"),
    ("consistency (score)", "Average of: share of pressure innings beating expectation, typical (median) pressure innings, minus bad-day rate"),
    ("median_pressure_innings", "Typical pressure innings value in runs vs expected (after pull toward league median)"),
    ("bad_day_pct", "% of pressure innings in the league's worst 20% (after pull toward league rate)"),
    ("pres_balls", "Balls faced/bowled in the top-20% pressure situations"),
    ("pressure_innings", "Innings with >=5 pressure balls faced (bat) or >=6 bowled (bowl)"),
    ("pres_SR / pres_xSR", "Strike rate in pressure balls vs what an average batter would make in the same situations"),
    ("pres_outs / pres_x_outs", "Dismissals in pressure balls vs expected"),
    ("pres_econ / pres_x_econ", "Economy in pressure balls vs expected; pres_wkts vs pres_x_wkts likewise"),
    ("normal_level", "Runs added per 100 balls vs average, in normal balls (after pull toward average)"),
    ("pressure_level", "Runs added per 100 balls vs average, in pressure balls (after pull toward player's normal level)"),
    ("rise", "pressure_level - normal_level"),
    ("consistency_pct", "Share of pressure innings where the player beat expectation (pulled toward league share)"),
    ("rank_5pct - rank_95pct", "90% range of rank when matches are re-drawn 300 times (bootstrap)"),
    ("pct_boot_top10", "% of re-draws in which the player made the top 10"),
    ("rank_top15pct_cut / rank_top25pct_cut", "Rank if 'pressure' is defined as top 15% / top 25% of balls instead of top 20%"),
    ("rank_IPL_only / rank_T20I_only", "Rank using only IPL / only T20I balls (min 60 pressure balls in that competition; blank = not enough)"),
    ("raw_pres_per100", "Raw pressure value per 100 balls before any pulling (for reference only)"),
    ("win_pressure", "Win impact under pressure: win-probability added per 100 pressure balls, after pull. NOT used in ranking (kept off by design)"),
    ("Pressure balls weighting", "Inside the pressure band, big-match balls count more: weight = sqrt(match importance / average)"),
    ("Conditions", "Expected values allow for venue scoring level that season and how everyone else in the match (both teams) batted/bowled"),
    ("Sheets 'OPP ...'", "TEST: pitch reading uses only the OPPOSITION's batters/bowlers instead of both teams"),
]
with pd.ExcelWriter(OUT / "step5_rankings.xlsx") as xw:
    pd.DataFrame(NOTES, columns=["column", "meaning"]).to_excel(xw, sheet_name="README", index=False)
    for role, lab in [("bat", "Batters"), ("bowl", "Bowlers")]:
        for th in [100, 150, 250]:
            pd.read_csv(OUT / f"step5_{role}_min{th}.csv").round(1).to_excel(xw, sheet_name=f"{lab} min{th}", index=False)
    for role, lab in [("bat", "Bat"), ("bowl", "Bowl")]:
        for th in [100, 150, 250]:
            f = OUT / f"step5_{role}_min{th}_pitchOpp.csv"
            if f.exists():
                pd.read_csv(f).round(1).to_excel(xw, sheet_name=f"OPP {lab} min{th}", index=False)
print("written", OUT / "step5_rankings.xlsx")
