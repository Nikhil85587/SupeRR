"""Step 2b: Match importance weight for each (match, team).
Weights are judgement calls, documented in the methodology report. Normal IPL match = 1.0.
Combination: product of the two highest weights (>=1) x the lowest weight (only if < 1).
If no match-type boost applies, the competition base (T20I 1.15, IPL 1.0, A-team 1.0) is in the pool.
A-team events: final value x 0.7.
Output: data/processed/match_importance.parquet, outputs/match_importance_summary.txt
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
W = dict(t20i_base=1.15, top3=1.4, rival=1.75, associate=0.9, t20i_final=2.0, super4=1.3, decider=1.5, must_win=1.65,
         ipl_q1=1.4, ipl_elim=1.7, ipl_q2=1.75, ipl_final=2.0, semi_final=1.75, do_or_die=1.5,
         dead_rubber=0.8, away=1.2, neutral=1.1, a_team_base=0.7)
NON_KO_CAP = 1.65  # max importance for any game that is not a final / IPL playoff / semi-final
RIVALS = [{"India", "Pakistan"}, {"Australia", "England"}, {"Bangladesh", "Sri Lanka"}]
COUNTRY = {"England": "United Kingdom", "Scotland": "United Kingdom", "United States of America": "United States",
           "United States Of America": "United States"}

m = pd.read_parquet(P / "matches_ext.parquet")
m["match_date"] = pd.to_datetime(m.match_date)
ctx = pd.read_parquet(P / "match_team_context.parquet")
m = m.sort_values(["comp_id", "match_date", "match_number"]).reset_index(drop=True)

flags = {}  # (match_id, team_id) -> list of (reason, weight)
def add(mid, tid, reason, w):
    flags.setdefault((mid, tid), []).append((reason, w))

def winner_side(r):
    if r.no_result or pd.isna(r.win_team_id):
        return None
    return r.win_team_id

# ---------- stage-based (finals, playoffs, Super 4, semis) ----------
for comp_id, g in m.groupby("comp_id"):
    g = g.sort_values(["match_date", "match_number"])
    comp = g.competition.iloc[0]
    name = g.comp_name.iloc[0]
    q_seen = 0
    for r in g.itertuples():
        teams = [r.home_team_id, r.away_team_id]
        stage = r.stage
        if name == "Mens Twenty20 Asia Cup 2022" and 7 <= r.match_number <= 12:
            stage = "Super 4"
        if comp == "IPL":
            if stage == "Qualifier":
                q_seen += 1
                tag = ("IPL Qualifier 1", W["ipl_q1"]) if q_seen == 1 else ("IPL Qualifier 2", W["ipl_q2"])
            else:
                tag = {"Eliminator": ("IPL Eliminator", W["ipl_elim"]), "Final": ("IPL Final", W["ipl_final"])}.get(stage)
        else:
            tag = {"Final": ("Final", W["t20i_final"]), "Super 4": ("Asia Cup Super 4", W["super4"]),
                   "Semi Final": ("Semi-final", W["semi_final"])}.get(stage)
        if tag:
            for t in teams:
                add(r.match_id, t, *tag)

# ---------- IPL points table: do-or-die and already-out ----------
for season, g in m[(m.competition == "IPL") & (m.stage == "League")].groupby("season"):
    g = g.sort_values(["match_date", "match_number"])
    teams = set(g.home_team_id) | set(g.away_team_id)
    pts = {t: 0 for t in teams}
    left = {t: ((g.home_team_id == t) | (g.away_team_id == t)).sum() for t in teams}
    for r in g.itertuples():
        for t, o in [(r.home_team_id, r.away_team_id), (r.away_team_id, r.home_team_id)]:
            others = sorted([pts[x] for x in teams if x != t], reverse=True)
            fourth = others[3]  # 4th-best of the other teams right now
            max_if_lose = pts[t] + 2 * (left[t] - 1)
            max_if_win_all = pts[t] + 2 * left[t]
            if max_if_win_all < fourth or max_if_win_all <= 12:
                add(r.match_id, t, "IPL already out (dead rubber)", W["dead_rubber"])
            elif max_if_lose <= max(fourth, 12):
                add(r.match_id, t, "IPL do-or-die (loss = out / near-out)", W["do_or_die"])
        w = winner_side(r)
        if w is None:
            if r.no_result:
                pts[r.home_team_id] += 1; pts[r.away_team_id] += 1
        else:
            pts[w] += 2
        left[r.home_team_id] -= 1; left[r.away_team_id] -= 1

# ---------- T20I bilateral series: decider / must-win / dead rubber ----------
bil = m[(m.competition == "T20I") & m.comp_name.str.contains("Series|series") & ~m.comp_name.str.contains("Tri")]
for comp_id, g in bil.groupby("comp_id"):
    g = g.sort_values(["match_date", "match_number"])
    n = len(g)
    if n < 2 or g[["home_team_id", "away_team_id"]].stack().nunique() != 2:
        continue
    wins = {}
    for i, r in enumerate(g.itertuples()):
        remaining = n - i  # including this one
        a, b_ = r.home_team_id, r.away_team_id
        wa, wb = wins.get(a, 0), wins.get(b_, 0)
        # dead rubber: the trailing side cannot even level the series by winning every remaining game
        if abs(wa - wb) > remaining:
            for t in (a, b_):
                add(r.match_id, t, "Dead rubber (series decided)", W["dead_rubber"])
        elif remaining == 1 and wa == wb and n >= 3:
            for t in (a, b_):
                add(r.match_id, t, "Series decider", W["decider"])
        else:
            for t, wt, wo in [(a, wa, wb), (b_, wb, wa)]:
                # losing this match means the team can no longer win or level the series
                if wo + 1 > wt + (remaining - 1):
                    add(r.match_id, t, "Must-win to save series", W["must_win"])
        w = winner_side(r)
        if w is not None:
            wins[w] = wins.get(w, 0) + 1

# ---------- opponent strength, rivalry, venue ----------
ctx_i = ctx.set_index(["match_id", "team_id"])
rows = []
for r in m.itertuples():
    for t, team, opp, opp_id in [(r.home_team_id, r.home_team, r.away_team, r.away_team_id),
                                 (r.away_team_id, r.away_team, r.home_team, r.home_team_id)]:
        comp = r.competition
        c = ctx_i.loc[(r.match_id, t)] if (r.match_id, t) in ctx_i.index else None
        if isinstance(c, pd.DataFrame):
            c = c.iloc[0]
        if comp == "T20I" and c is not None:
            if c.opp_rank <= 3:
                add(r.match_id, t, f"Opponent ranked top-3 (#{int(c.opp_rank)})", W["top3"])
            if c.opp_is_associate:
                add(r.match_id, t, "Opponent is an associate", W["associate"])
        strip = lambda s: s[:-2] if isinstance(s, str) and s.endswith(" A") else s
        if {strip(team), strip(opp)} in RIVALS and comp != "IPL":
            add(r.match_id, t, "Rivalry", W["rival"])
        # venue
        if comp == "IPL":
            if r.season == "2022" or str(r.season) == "2022":
                venue = "neutral"
            elif r.stage == "League":
                venue = "home" if t == r.home_team_id else "away"
            else:  # playoffs: home only if at the team's main home ground that season
                lg = m[(m.competition == "IPL") & (m.season == r.season) & (m.stage == "League")]
                home_v = lambda tid: lg[lg.home_team_id == tid].venue_id.mode().iloc[0]
                if r.venue_id == home_v(t):
                    venue = "home"
                elif r.venue_id == home_v(opp_id):
                    venue = "away"
                else:
                    venue = "neutral"
        else:
            vc = r.venue_country
            mine, theirs = COUNTRY.get(strip(team), strip(team)), COUNTRY.get(strip(opp), strip(opp))
            venue = "home" if vc == mine else ("away" if vc == theirs else "neutral")
        if venue == "away":
            add(r.match_id, t, "Away", W["away"])
        elif venue == "neutral":
            add(r.match_id, t, "Neutral venue", W["neutral"])

        f = flags.get((r.match_id, t), [])
        match_type_boost = any(w > 1 for reason, w in f if reason not in ("Away", "Neutral venue"))
        base = W["t20i_base"] if comp == "T20I" else 1.0
        pool = [w for _, w in f] + ([] if match_type_boost else [base])
        boosts = sorted([w for w in pool if w >= 1], reverse=True)[:2]
        discounts = [w for w in pool if w < 1]
        weight = float(np.prod(boosts) if boosts else 1.0) * (min(discounts) if discounts else 1.0)
        # cap (decided after re-reading the brief): a non-knockout game can never carry more weight than the
        # lowest IPL playoff (1.65), so bilateral must-wins/rivalries cannot outrank IPL playoffs or finals
        is_knockout = any(k in reason for reason, _ in f for k in ("Final", "Qualifier", "Eliminator", "Semi-final"))
        if not is_knockout:
            weight = min(weight, NON_KO_CAP)
        if comp == "A-team":
            weight *= W["a_team_base"]
        rows.append(dict(match_id=r.match_id, team_id=t, team=team, opponent=opp, competition=comp,
                         comp_name=r.comp_name, match_date=r.match_date, stage=r.stage, venue=venue,
                         importance=round(weight, 4),
                         reasons="; ".join(f"{a} ({w})" for a, w in f) or "Normal"))

imp = pd.DataFrame(rows)
imp.to_parquet(P / "match_importance.parquet", index=False)

with open(ROOT / "outputs/match_importance_summary.txt", "w", encoding="utf-8") as fh:
    def out(*a):
        print(*a); print(*a, file=fh)
    out("Team-matches:", len(imp))
    out("\nImportance distribution by competition:\n", imp.groupby("competition").importance.describe().round(2).to_string())
    flat = imp.reasons.str.split("; ").explode().str.replace(r" \(.*\)$", "", regex=True).str.replace(r"#\d", "#", regex=True)
    out("\nHow often each reason applies (team-matches):\n", flat.value_counts().to_string())
    out("\nTop 15 most important team-matches:\n",
        imp.sort_values("importance", ascending=False).head(15)[["match_date", "team", "opponent", "comp_name", "importance", "reasons"]].to_string(index=False))
    s = imp[(imp.comp_name == "IPL 2024")].sort_values("match_date")
    out("\nIPL 2024 do-or-die / already-out flags:\n",
        s[s.reasons.str.contains("do-or-die|already out")][["match_date", "team", "opponent", "reasons"]].to_string(index=False))
