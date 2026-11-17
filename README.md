# Most Reliable Players Under Pressure — SupeRR Selector Season 2, Round 1

Ranks the 10 best batters and 10 best bowlers under pressure in IPL and T20I cricket (2022–2026), from the
supplied ball-by-ball data.

## Run it

1. Install Python 3.11 and the packages: `pip install -r requirements.txt`
2. Put the six supplied competition CSV files (`ball_by_ball.csv`, `matches.csv`, `competitions.csv`, `teams.csv`,
   `playing_eleven.csv`, `competition_team_player_mapping.csv`) in a folder called `Updated Data/` at the project root.
   They are the organisers' data and are not included in this repository.
3. Run everything:

```
python -P -E run_all.py
```

This downloads the public Cricsheet men's T20I results (used only for team ratings and form) if they are missing,
then runs the nine steps below in order (about 15 minutes of compute). `data/` holds downloads and intermediate
files; `outputs/` holds the results, which are committed here so they can be read without re-running.

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `src/01_cricsheet_results.py` | Reads every men's T20I result from Cricsheet |
| 2 | `src/02_ball_states.py` | Rebuilds score, target, required rate, wickets and balls left before every ball (`runs` is bat runs only; the true extra runs come from `extras1_runs` + `extras_2_runs`, which the updated data set records) |
| 3 | `src/03_team_strength.py` | Rebuilds ICC T20I team ratings by date with the ICC's method, last-5 form, pre-match win chances |
| 4 | `src/04_match_importance.py` | Match-importance weight per team per match (finals, playoffs, deciders, rivalries, venue …) |
| 5 | `src/05_pressure.py` | Win-probability model, leverage of every ball, pressure score (60% situation, 40% importance), validation |
| 6 | `src/06_player_values.py` | Expected outcome of every ball (incl. conditions and opponent quality) and runs added |
| 7 | `src/07_ratings.py` | Usual vs pressure level, pull toward usual level, consistency, rise (all seasons weighted equally), rankings at 100/150/250 pressure balls, 300 match re-draws |
| 8 | `src/08_review_workbook.py` | `outputs/step5_rankings.xlsx` (all lists + column guide) |
| 9 | `src/09_report_tables.py` | Tables used in the methodology report |

## Main outputs

- `outputs/step5_bat_min150.csv`, `outputs/step5_bowl_min150.csv` — the submitted rankings (minimum 150 pressure balls).
- `outputs/step5_rankings.xlsx` — every list with its reliability columns.
- `outputs/pressure_validation.txt` — win-model accuracy and the check that pressure balls play differently.

## What changed in the updated data set (2026-11-15)

Only `ball_by_ball.csv` differs from the first delivery (the other CSVs are byte-identical):

- `extras1_runs` is now the **actual runs of the extra** (wide 1-5, no-ball 1, bye / leg-bye 1-4, penalty 5). Before, it held an
  extras *type code* and every extra had to be counted as 1 run. Rebuilt first-innings totals now match the official run margin in
  99.7% of matches (before: 32%), so every score, target and required rate is exact.
- "AHT Donald" is now "Aneurin Donald" (same player ID 4458).
- `runs`, `extras_2_runs` and every other column are unchanged.

## Optional switches (checks, not the submitted ranking)

- `src/06_player_values.py --no-quality` — no opponent-quality adjustment; `--pitch opp` — read the pitch from the opposition only.
- `src/07_ratings.py --recency` (old season weights 2022 0.4 … 2026 1.0), `--cons simple`, `--rise positive`, `--win-share 0.3`.
