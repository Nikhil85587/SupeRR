"""Step 10: build the interactive dashboard as one self-contained web page: docs/index.html

The page shows every weight used in the model (match importance, pressure mix, final-score mix, pull strengths) with the
reason for each, an example chase, the pressure check, and the rankings with their reliability.
All data is embedded in the page, so it can be opened straight from disk or served by GitHub Pages.
Inputs: outputs/step5_*_min150.csv, outputs/step5_notes.txt, outputs/report_example_chase.csv,
        data/processed/match_importance.parquet, src/dashboard_template.html
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)

# ---------------- match-importance weights with their reasons ----------------
imp = pd.read_parquet(ROOT / "data/processed/match_importance.parquet")
imp["match_date"] = pd.to_datetime(imp.match_date).dt.strftime("%Y-%m-%d")
fl = imp.reasons.map(lambda s: [] if s == "Normal" else [re.sub(r" \([\d.]+\)$", "", x) for x in s.split("; ")])
def has(prefix): return fl.map(lambda l: any(x.startswith(prefix) for x in l))
BOOSTS = ("Final", "IPL Final", "IPL Qualifier", "IPL Eliminator", "Semi-final", "Asia Cup Super 4", "Series decider",
          "Must-win", "IPL do-or-die", "Rivalry", "Opponent ranked top-3")
no_boost = ~fl.map(lambda l: any(x.startswith(BOOSTS) for x in l))

W = [  # weights must match W in src/04_match_importance.py
 ("ipl_base", "Normal IPL game", 1.0, "Baseline", "IPL", (imp.competition == "IPL") & no_boost,
  "The yardstick. An ordinary IPL league game counts 1.0 and every other weight is measured against it."),
 ("t20i_base", "Normal T20I", 1.15, "Competition", "T20I", (imp.competition == "T20I") & no_boost,
  "Playing for your country carries more scrutiny than an ordinary IPL league game. Used only when no bigger match-type reason applies."),
 ("a_team", "Emerging / A-team event", 0.7, "Competition", "A-team", imp.competition == "A-team",
  "Emerging Teams Asia Cup 2024 and Asia Cup Rising Stars 2025: a lower level with less at stake. Multiplies the final weight."),
 ("t20i_final", "T20I final", 2.0, "Knockout", "T20I", has("Final") & (imp.competition != "IPL"),
  "One game decides the trophy. Ties with the IPL Final as the biggest single factor."),
 ("ipl_final", "IPL Final", 2.0, "Knockout", "IPL", has("IPL Final"), "One game decides the IPL title."),
 ("ipl_q2", "IPL Qualifier 2", 1.75, "Knockout", "IPL", has("IPL Qualifier 2"), "Loser is out, winner goes to the final."),
 ("semi", "Semi-final", 1.75, "Knockout", "A-team", has("Semi-final"),
  "Loser is out. Treated like IPL Qualifier 2 (only the A-team events have semi-finals in this data)."),
 ("ipl_elim", "IPL Eliminator", 1.7, "Knockout", "IPL", has("IPL Eliminator"), "Loser is out of the tournament."),
 ("ipl_q1", "IPL Qualifier 1", 1.4, "Knockout", "IPL", has("IPL Qualifier 1"),
  "Winner reaches the final, but the loser still gets a second chance in Qualifier 2, so less is at stake than in the other knockouts."),
 ("rival", "Rivalry (Ind-Pak, Aus-Eng, Ban-SL)", 1.75, "Opponent", "T20I", has("Rivalry"),
  "Games with heavy public and media weight. Only in internationals, never in the IPL."),
 ("top3", "Opponent ranked top-3", 1.4, "Opponent", "T20I", has("Opponent ranked top-3"),
  "Facing one of the three best sides on the ICC T20I ranking for that day (rebuilt with the ICC's own method)."),
 ("associate", "Opponent is an associate nation", 0.9, "Opponent", "T20I", has("Opponent is an associate"),
  "Against a non-Test-playing nation a win is expected, so a little less is at stake."),
 ("must_win", "Must-win to save the series", 1.65, "Series / table", "T20I", has("Must-win"),
  "The trailing team loses the series if it loses this game."),
 ("decider", "Series decider", 1.5, "Series / table", "T20I", has("Series decider"),
  "Series level going into the last game of a series of 3 or more games: winner takes the series."),
 ("do_or_die", "IPL do-or-die league game", 1.5, "Series / table", "IPL", has("IPL do-or-die"),
  "A loss would leave the team unable to reach the current 4th-place points (or 12 points)."),
 ("super4", "Asia Cup Super 4", 1.3, "Series / table", "T20I", has("Asia Cup Super 4"),
  "Every Super 4 game counts towards a place in the Asia Cup final."),
 ("dead_rubber", "Dead rubber / already out", 0.8, "Series / table", "Both", has("Dead rubber") | has("IPL already out"),
  "Nothing left to play for: the series is already decided, or the IPL team can no longer qualify."),
 ("away", "Away", 1.2, "Venue", "Both", has("Away"), "Opponent's country or home ground: hostile crowd and unfamiliar conditions."),
 ("neutral", "Neutral venue", 1.1, "Venue", "Both", has("Neutral venue"),
  "Neither side at home. All of IPL 2022 was played at neutral venues."),
]
data = {}
data["match_weights"] = pd.DataFrame([dict(key=k, factor=f, weight=w, group=g, competition=c, team_matches=int(m.sum()), reason=r)
                                      for k, f, w, g, c, m, r in W])
data["team_matches"] = imp[["match_date", "team", "opponent", "competition", "comp_name", "stage", "venue", "importance", "reasons"]] \
    .sort_values(["match_date", "team"])

# ---------------- model settings and pull strengths ----------------
data["model_settings"] = pd.DataFrame([dict(key=a, stage=b, setting=c, percent=d, reason=e) for a, b, c, d, e in [
 ("situation", "Pressure score", "Game situation (leverage)", 60,
  "How much the next ball can swing the win chance, from a win-probability model (score, wickets, balls left, target and required rate, pre-match odds). Situation comes first: a tight finish is pressure in any match."),
 ("importance", "Pressure score", "Match importance", 40,
  "The match weight from the table above. A final or a must-win game adds pressure even before the first ball."),
 ("band", "Pressure balls", "Top share of balls that count as pressure", 20,
  "A ball counts as a pressure ball if its pressure score is in the top 20% of all balls. 15% and 25% are run as checks."),
 ("pres_level", "Final score", "Pressure level", 62.5,
  "Runs added per 100 pressure balls compared with an average player in the same spots. The main thing the brief asks for: who performs best under pressure."),
 ("consistency", "Final score", "Consistency", 22.5,
  "Average of three things: share of pressure innings that beat expectation, the typical (median) pressure innings, and fewer bad days. The brief asks who delivers most consistently."),
 ("rise", "Final score", "Rise vs own usual level", 15,
  "Pressure level minus the player's usual (normal-ball) level. Rewards players who lift under pressure and marks down those who dip, as the brief asks."),
]])
notes = (ROOT / "outputs/step5_notes.txt").read_text(encoding="utf-8")
rows = []
for role, part, lab, why in [
        ("bat", "bat_score_part", "Batting: scoring rate", "Strike rate repeats well from one half of a career to the other, so a light pull is enough."),
        ("bat", "bat_out_part", "Batting: getting out", "Dismissal rate is noisier, so it is pulled harder towards the usual level."),
        ("bowl", "bowl_econ_part", "Bowling: economy", "Economy is the most repeatable part of all, so it gets the lightest pull."),
        ("bowl", "bowl_wkt_part", "Bowling: wickets", "Wicket-taking barely repeats in samples this size, so a few lucky wickets move a bowler very little.")]:
    m = re.search(rf"\[{role}\] {part}: split-half r ([\d.]+) \| pull strength: normal level k=(\d+)", notes)
    rows.append(dict(part=lab, role=role, split_half_r=float(m[1]), k_balls=int(m[2]), reason=why))
data["pull_strength"] = pd.DataFrame(rows)

# ---------------- rankings, example chase, validation ----------------
tail = ["normal_level", "pressure_level", "rise", "consistency_pct", "median_pressure_innings", "bad_day_pct", "final_score",
        "rank_5pct", "rank_95pct", "pct_boot_top10", "rank_top15pct_cut", "rank_top25pct_cut", "rank_IPL_only", "rank_T20I_only"]
for role, mid in [("bat", ["pres_SR", "pres_xSR", "pres_outs", "pres_x_outs"]),
                  ("bowl", ["pres_econ", "pres_x_econ", "pres_wkts", "pres_x_wkts"])]:
    d = pd.read_csv(ROOT / f"outputs/step5_{role}_min150.csv")
    d = d[["rank", "player", "country", "ipl_teams", "pres_balls", "pressure_innings"] + mid + tail]
    d["country"] = d.country.fillna("")
    data[f"rank_{role}"] = d.round(4)

# ---------------- alternative settings: rank of each top-15 player under each one ----------------
for role in ["bat", "bowl"]:
    main = pd.read_csv(ROOT / f"outputs/step5_{role}_min150.csv")
    alt = {th: pd.read_csv(ROOT / f"outputs/step5_{role}_min{th}.csv").set_index("player")["rank"] for th in (100, 250)}
    t = main[main["rank"] <= 15][["rank", "player", "rank_top15pct_cut", "rank_top25pct_cut", "rank_IPL_only", "rank_T20I_only"]].copy()
    t.insert(2, "min100", t.player.map(alt[100]))
    t.insert(3, "min250", t.player.map(alt[250]))
    ranks = t.drop(columns=["player"])
    t["settings_in_top10"] = (ranks <= 10).sum(axis=1).astype(int)
    t["settings_qualified"] = ranks.notna().sum(axis=1).astype(int)
    data[f"robust_{role}"] = t.astype({c: "Int64" for c in ranks.columns})

# ---------------- data coverage ----------------
facts = (ROOT / "outputs/report_facts.txt").read_text(encoding="utf-8")
num = lambda pat: int(re.search(pat, facts)[1].replace(",", ""))
share = re.search(r"Pressure-ball share by competition \(batting\): \{'T20I': ([\d.]+).*?all balls: \{'T20I': ([\d.]+)", facts)
dates = pd.to_datetime(imp.match_date)
data["coverage"] = pd.DataFrame([dict(key=k, label=l, value=v, note=n) for k, l, v, n in [
    ("period", "Period", f"{dates.min():%b %Y} – {dates.max():%b %Y}", "IPL 2022–2026 seasons and men's T20Is in the same years."),
    ("matches", "Matches with ball-by-ball data", num(r"matches (\d+)"), "883 matches are listed; 16 have no ball-by-ball data."),
    ("balls", "Balls", num(r"Balls ([\d,]+)"), "Every delivery, with score, wickets, target and required rate rebuilt before each ball."),
    ("ipl", "IPL matches", num(r"IPL matches (\d+)"), "Five seasons, all playoffs included."),
    ("t20i", "T20I matches", num(r"T20I (\d+)"), "Bilateral series, Asia Cup 2022 and 2025, and tri-series. No T20 World Cups in the data."),
    ("ateam", "Emerging / A-team matches", num(r"A-team (\d+)"), "Emerging Teams Asia Cup 2024 and Asia Cup Rising Stars 2025."),
    ("t20i_share", "Pressure balls from T20Is", f"{round(100 * float(share[1]))}%",
     f"T20Is are {round(100 * float(share[2]))}% of all balls; bigger matches push more of them into the pressure band."),
    ("bat_pool", "Batters ranked", len(data["rank_bat"]), "At least 150 pressure balls faced."),
    ("bowl_pool", "Bowlers ranked", len(data["rank_bowl"]), "At least 150 pressure balls bowled."),
    ("external", "Outside data", "Cricsheet", "Men's T20I results from 2018, used only to rebuild ICC team ratings and recent form. Cricsheet withholds Afghanistan's matches, so their rating is partly rebuilt."),
]])

ch = pd.read_csv(ROOT / "outputs/report_example_chase.csv")
ch["ball"] = (ch.overid.astype(int) - 1).astype(str) + "." + ch.over_ball_num.astype(str)
ch["pressure_ball"] = ch.pct_bat >= 80
data["example_chase"] = ch[["innings_ball_num", "ball", "legal", "runs", "total_runs", "is_wicket", "runs_needed", "balls_left",
                            "wp_bat", "LI", "pct_bat", "pressure_ball", "striker_name", "bowler_name"]].round(4)

val = (ROOT / "outputs/pressure_validation.txt").read_text(encoding="utf-8")
blk = val[val.index("LIKE-FOR-LIKE"):]
lo = [float(x) for x in re.search(r"Low pressure\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", blk).groups()]
hi = [float(x) for x in re.search(r"High pressure\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", blk).groups()]
ch_ = [float(x) for x in re.search(r"Change\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", blk).groups()]
data["validation"] = pd.DataFrame([dict(measure=m, low=lo[i], high=hi[i], change_pct=round(100 * ch_[i], 1))
                                   for i, m in enumerate(["Runs per ball", "Dot-ball rate", "Boundary rate", "Wicket rate"])])

# ---------------- page ----------------
template = (ROOT / "src/dashboard_template.html").read_text(encoding="utf-8")
csv = {k: v.to_csv(index=False, lineterminator="\n") for k, v in data.items()}
assert not any("</" in t for t in csv.values()), "data text would end the embedded <script> block early"
embedded = "\n".join(f'<script type="text/csv" id="csv-{k}">\n{t}</script>' for k, t in csv.items())

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pressure Performers</title>
<meta name="description" content="Who performs best under pressure in IPL and T20I cricket, 2022-2026: every weight, its reason, and the rankings.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400&display=swap">
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<style>
:root {
  --color-bg: #f7f8fa; --color-panel: #ffffff; --color-fg: #17202e; --color-fg-muted: #5b6577; --color-border-line: #dfe3ea;
  --cds-text-secondary: #5b6577; --cds-border: #d3d8e1; --cds-surface-popover: #ffffff; --cds-shadow-popover: 0 6px 24px rgba(20, 30, 50, .14);
  --cds-chart-grid: #e8ebf0; --cds-chart-axis: #c6ccd6; --cds-chart-muted: #c3c9d3; --cds-chart-reference: #8a93a3; --cds-chart-status-critical: #c8372d;
  --c1: #2f6fdb; --c2: #e0782f; --c3: #179a83;
  --font-anthropic-sans: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-anthropic-serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --cds-font-size-caption: 12.5px; --cds-font-size-body: 15px; --cds-font-size-heading: 21px; --cds-font-size-title: 34px; --cds-font-weight-medium: 600;
  --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 14px; --cds-gap-lg: 20px; --cds-gap-xl: 40px; --cds-pad-md: 12px; --cds-pad-lg: 20px; --cds-radius: 10px;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root {
    --color-bg: #10141b; --color-panel: #171c25; --color-fg: #e7ebf2; --color-fg-muted: #9aa4b5; --color-border-line: #2a3240;
    --cds-text-secondary: #9aa4b5; --cds-border: #323b4a; --cds-surface-popover: #1d2330; --cds-shadow-popover: 0 6px 24px rgba(0, 0, 0, .5);
    --cds-chart-grid: #232a36; --cds-chart-axis: #3a4352; --cds-chart-muted: #3d4655; --cds-chart-reference: #7d8798; --cds-chart-status-critical: #ef6b5f;
    --c1: #6b9cf0; --c2: #f0975a; --c3: #3cc4aa;
    color-scheme: dark;
  }
}
html, body { margin: 0; background: var(--color-bg); color: var(--color-fg); font-family: var(--font-anthropic-sans); font-size: 15px; }
.rr-page { max-width: 1080px; margin: 0 auto; padding: 32px max(16px, env(safe-area-inset-right, 0px)) 64px max(16px, env(safe-area-inset-left, 0px)); }
.rr-foot { margin-top: 48px; font-size: 12.5px; color: var(--color-fg-muted); }
.rr-foot a { color: inherit; }
.dash-select { font: inherit; font-size: 14px; color: var(--color-fg); background-color: var(--color-panel); border: 1px solid var(--color-border-line); border-radius: 8px; padding: 7px 10px; }
[hidden] { display: none !important; }
@media (max-width: 600px) { :root { --cds-font-size-title: 27px; --cds-gap-xl: 30px; } }
</style>
</head>
<body>
<div class="rr-page">
"""

SHIM = """<script>
// Minimal stand-in for the dashboard runtime: data from the embedded CSVs, calculations, and redrawn drawings.
window.dash = (() => {
  const raw = {}, calcs = {}, cache = {}, draws = new Map();
  document.querySelectorAll('script[type="text/csv"]').forEach((s) => { raw[s.id.slice(4)] = d3.csvParse(s.textContent.trim()); });
  const data = (id) => {
    if (raw[id]) return { status: 'ok', data: raw[id] };
    if (calcs[id]) {
      if (!(id in cache)) cache[id] = calcs[id].fn(...calcs[id].inputs.map((i) => data(i).data));
      return { status: 'ok', data: cache[id] };
    }
    return { status: 'missing', data: [] };
  };
  const runOne = (name) => {
    const fn = draws.get(name); if (!fn) return;
    document.querySelectorAll(`[data-draw="${name}"]`).forEach((el) => { if (!el.closest('[hidden]')) fn(el); });
  };
  const runAll = () => draws.forEach((_, name) => runOne(name));
  let started = false;
  return {
    data,
    calc: (id, spec) => { calcs[id] = spec; },
    loader: () => {},
    onData: (fn) => fn(),
    draw: (name, fn) => { if (fn) { draws.set(name, fn); if (started) runOne(name); } else runOne(name); },
    colors: ['var(--c1)', 'var(--c2)', 'var(--c3)'],
    params: () => ({}), setParams: () => {}, resetParams: () => {}, setLink: () => {}, refresh: () => {},
    start: () => {
      started = true; runAll();
      let t; window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(runAll, 150); });
      // redraw when a section is shown (tooltips also toggle `hidden`, so ignore them)
      new MutationObserver((ms) => { if (ms.some((m) => !m.target.classList.contains('rr-tip'))) runAll(); })
        .observe(document.body, { attributes: true, attributeFilter: ['hidden'], subtree: true });
      if (document.fonts) document.fonts.ready.then(runAll);
    },
  };
})();
</script>
"""

FOOT = """
<p class="rr-foot">Source code, data pipeline and full results:
<a href="https://github.com/Nikhil85587/SupeRR">github.com/Nikhil85587/SupeRR</a>. Rankings use players with at least 150 pressure balls.</p>
</div>
<script>dash.start();</script>
</body>
</html>
"""

page = HEAD + embedded + "\n" + SHIM + template + FOOT
(OUT / "index.html").write_text(page, encoding="utf-8")
(OUT / ".nojekyll").write_text("", encoding="utf-8")
print("written", OUT / "index.html", f"{len(page) / 1024:.0f} KB")
