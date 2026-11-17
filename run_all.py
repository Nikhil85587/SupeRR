"""Rebuild every result from scratch: python -P -E run_all.py  (about 15 minutes of compute; step 06 alone is ~75 s).

Reads the supplied CSVs from the "Updated Data" folder (the original files in the project root are never used or modified).
"""
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ZIP = ROOT / "data/external/cricsheet_t20i/t20s_male_json.zip"
STEPS = ["01_cricsheet_results", "02_ball_states", "03_team_strength", "04_match_importance", "05_pressure",
         "06_player_values", "07_ratings", "08_review_workbook", "09_report_tables"]

if not (ROOT / "Updated Data" / "ball_by_ball.csv").exists():
    sys.exit("Put the supplied CSV files (ball_by_ball, matches, competitions, teams ...) in the 'Updated Data' folder.")
for d in ["data/external/cricsheet_t20i", "data/processed", "outputs"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)
if not ZIP.exists():
    print("Downloading Cricsheet men's T20I results ...")
    urllib.request.urlretrieve("https://cricsheet.org/downloads/t20s_male_json.zip", ZIP)

for step in STEPS:
    print(f"\n=== {step} ===", flush=True)
    r = subprocess.run([sys.executable, "-P", "-E", str(ROOT / "src" / f"{step}.py")], cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"{step} failed (exit code {r.returncode})")
print("\nDone. Rankings: outputs/step5_rankings.xlsx; report tables: outputs/report_*.csv")
