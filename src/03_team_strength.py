"""Step 2a: Team strength before every match.
 - T20I: ICC-rule rating rebuild from all men's T20Is (Cricsheet + our matches.csv to fill Afghanistan, which Cricsheet withholds).
 - Last-5 form (win share of last 5 completed matches) for T20I teams and IPL franchises.
 - Pre-match win probability p0 for each side (logistic on rating gap / form), used by the win-probability model.
Outputs: data/processed/icc_ratings_daily.parquet, data/processed/match_team_context.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
FULL_MEMBERS = {"India", "Australia", "England", "New Zealand", "South Africa", "Pakistan", "Sri Lanka",
                "Bangladesh", "West Indies", "Afghanistan", "Ireland", "Zimbabwe"}
NAME_FIX = {"United States Of America": "United States of America"}
DEFAULT_FM, DEFAULT_ASSOC = 100.0, 50.0  # ICC-style base ratings for teams with no matches in the window


def default_rating(team):
    return DEFAULT_FM if team in FULL_MEMBERS else DEFAULT_ASSOC

# ---------------- 1. all T20I results ----------------
cs = pd.read_csv(P / "t20i_results.csv", parse_dates=["date"])
cs = cs[cs.result != "no result"].copy()
cs["outcome1"] = np.where(cs.result == "tie", 0.5, (cs.winner == cs.team1).astype(float))
cs = cs[["date", "team1", "team2", "outcome1"]].assign(src="cricsheet")

m = pd.read_parquet(P / "matches_ext.parquet")
m["match_date"] = pd.to_datetime(m.match_date)
for col in ["home_team", "away_team"]:
    m[col] = m[col].replace(NAME_FIX)
ours = m[(m.competition == "T20I") & ~m.no_result].copy()
ours["outcome1"] = np.where(ours.tied & ours.win_team_id.isna(), 0.5, (ours.win_team_id == ours.home_team_id).astype(float))
key = lambda d, a, b_: d.dt.strftime("%Y-%m-%d") + "|" + pd.Series(np.sort(np.c_[a, b_], axis=1).tolist(), index=a.index).str.join("|")
cs_keys = set(key(cs.date, cs.team1, cs.team2))
# allow 1-day date differences (time zones)
cs_keys |= set(key(cs.date + pd.Timedelta(days=1), cs.team1, cs.team2)) | set(key(cs.date - pd.Timedelta(days=1), cs.team1, cs.team2))
ours["k"] = key(ours.match_date, ours.home_team, ours.away_team)
add = ours[~ours.k.isin(cs_keys)]
print(f"our T20Is: {len(ours)}, found in Cricsheet: {len(ours) - len(add)}, added from our data: {len(add)}")
print("  added teams:", add[["home_team", "away_team"]].stack().value_counts().head(6).to_dict())
res = pd.concat([cs, add.rename(columns={"match_date": "date", "home_team": "team1", "away_team": "team2"})
                 [["date", "team1", "team2", "outcome1"]].assign(src="ours")]).sort_values("date").reset_index(drop=True)


# ---------------- 2. ICC rating rebuild ----------------
def update_year(d):  # year of the most recent annual update (1 May)
    return d.year if (d.month, d.day) >= (5, 1) else d.year - 1


hist = {}  # team -> list of (date, points, )


def rating(team, d):
    h = hist.get(team)
    if not h:
        return default_rating(team), 0
    y = update_year(d)
    start, full_from = pd.Timestamp(y - 3, 5, 1), pd.Timestamp(y - 1, 5, 1)
    pts = wts = n = 0.0
    for dt, p in reversed(h):
        if dt >= d:  # only matches before date d (no look-ahead)
            continue
        if dt < start:
            break
        w = 1.0 if dt >= full_from else 0.5
        pts += w * p; wts += w; n += 1
    return (pts / wts, int(n)) if wts else (default_rating(team), 0)


def icc_points(r_own, r_opp, result):  # result: 1 win, 0.5 tie, 0 loss
    if abs(r_own - r_opp) < 40:
        return r_opp + {1: 50, 0.5: 0, 0: -50}[result]
    if r_own > r_opp:  # stronger side
        return r_own + {1: 10, 0.5: -40, 0: -90}[result]
    return r_own + {1: 90, 0.5: 40, 0: -10}[result]


pre = []
for row in res.itertuples():
    r1, n1 = rating(row.team1, row.date)
    r2, n2 = rating(row.team2, row.date)
    o = row.outcome1
    pre.append((r1, r2, n1, n2))
    hist.setdefault(row.team1, []).append((row.date, icc_points(r1, r2, o)))
    hist.setdefault(row.team2, []).append((row.date, icc_points(r2, r1, 1 - o if o != 0.5 else 0.5)))
res[["r1", "r2", "n1", "n2"]] = pre


def table(d, min_matches=8):
    rows = [(t, *rating(t, d)) for t in hist]
    df = pd.DataFrame(rows, columns=["team", "rating", "matches"])
    df = df[df.matches >= min_matches].sort_values("rating", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


# snapshots for validation
snaps = {}
for d in ["2022-05-05", "2023-05-05", "2024-05-05", "2025-05-05", "2026-05-05", "2026-11-01"]:
    snaps[d] = table(pd.Timestamp(d)).head(10)
    print(d, " | ".join(f"{r.team} {r.rating:.0f}" for r in snaps[d].itertuples()))
pd.concat({k: v for k, v in snaps.items()}).to_csv(ROOT / "outputs/icc_rebuild_snapshots.csv")

# ---------------- 3. per-match context for our matches ----------------
def last5(team_results, team, d):
    h = team_results.get(team, [])
    past = [o for dt, o in h if dt < d][-5:]
    return (np.mean(past) if past else np.nan), len(past)


t20_hist = {}
for row in res.itertuples():
    t20_hist.setdefault(row.team1, []).append((row.date, row.outcome1))
    t20_hist.setdefault(row.team2, []).append((row.date, 1 - row.outcome1))

# IPL / A-team form from our own data (IPL is complete in our data)
dom = m[(m.competition != "T20I") & ~m.no_result].sort_values("match_date")
dom_hist = {}
for row in dom.itertuples():
    o = 0.5 if (row.tied and pd.isna(row.win_team_id)) else float(row.win_team_id == row.home_team_id)
    dom_hist.setdefault(row.home_team, []).append((row.match_date, o))
    dom_hist.setdefault(row.away_team, []).append((row.match_date, 1 - o))

rows = []
for mt in m.itertuples():
    d = mt.match_date
    for side, team, tid, opp in [("home", mt.home_team, mt.home_team_id, mt.away_team),
                                 ("away", mt.away_team, mt.away_team_id, mt.home_team)]:
        rec = dict(match_id=mt.match_id, team_id=tid, team=team, opponent=opp, side=side, competition=mt.competition)
        if mt.competition == "T20I":
            tab = table(d)
            rk = dict(zip(tab.team, tab["rank"]))
            rec["rating"], _ = rating(team, d)
            rec["opp_rating"], _ = rating(opp, d)
            rec["rank"], rec["opp_rank"] = rk.get(team, np.nan), rk.get(opp, np.nan)
            rec["form5"], rec["form_n"] = last5(t20_hist, team, d)
            rec["opp_form5"], _ = last5(t20_hist, opp, d)
        else:
            rec["form5"], rec["form_n"] = last5(dom_hist, team, d)
            rec["opp_form5"], _ = last5(dom_hist, opp, d)
        rec["opp_is_associate"] = (mt.competition == "T20I") and (opp not in FULL_MEMBERS)
        rows.append(rec)
ctx = pd.DataFrame(rows)

# ---------------- 4. pre-match win probability p0 ----------------
# T20I: logistic on rating gap, fitted on all T20Is 2019+ (both teams rated)
fit = res[(res.date >= "2019-01-01") & (res.n1 >= 5) & (res.n2 >= 5) & (res.outcome1 != 0.5)]
lr = LogisticRegression().fit(((fit.r1 - fit.r2) / 100).to_frame("g"), fit.outcome1)
print(f"T20I p0 model: coef per 100 rating pts = {lr.coef_[0][0]:.2f}  (n={len(fit)})")
gap = ((ctx.rating - ctx.opp_rating) / 100).fillna(0).to_frame("g")
ctx["p0"] = np.where(ctx.competition == "T20I", lr.predict_proba(gap)[:, 1], np.nan)
# IPL / A-team: logistic on last-5 form difference, fitted on our IPL matches
dom_ctx = ctx[(ctx.competition != "T20I") & (ctx.side == "home")].merge(m[["match_id", "win_team_id", "no_result"]], on="match_id")
dom_ctx = dom_ctx[~dom_ctx.no_result & dom_ctx.win_team_id.notna()]
X = (dom_ctx.form5.fillna(0.5) - dom_ctx.opp_form5.fillna(0.5)).to_frame("f")
lr2 = LogisticRegression().fit(X, (dom_ctx.win_team_id == dom_ctx.team_id).astype(int))
print(f"IPL p0 model: coef on form gap = {lr2.coef_[0][0]:.2f}  (weak signal expected)")
Xa = (ctx.form5.fillna(0.5) - ctx.opp_form5.fillna(0.5)).to_frame("f")
ctx.loc[ctx.competition != "T20I", "p0"] = lr2.predict_proba(Xa)[ctx.competition.values != "T20I", 1]

ctx.to_parquet(P / "match_team_context.parquet", index=False)
res.to_parquet(P / "t20i_all_results_rated.parquet", index=False)
print("context rows:", len(ctx), "| T20I sides facing a top-3 team:", int((ctx.opp_rank <= 3).sum()))
