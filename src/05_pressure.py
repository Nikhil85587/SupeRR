"""Step 3: Win-probability model -> leverage of every ball -> pressure score.

Win probability (batting side) = logistic regression on cricket features, learnt from our own balls
(chosen over tree models: same or better accuracy, and it moves smoothly ball to ball - trees gave flat steps).
  1st innings: score, wickets down, balls left, wickets-in-hand x balls-left, sqrt(resources), pre-match odds
  (current run rate was dropped: it swings wildly in the first over and made powerplay balls look high-leverage)
  2nd innings: runs needed, balls left, wickets left, required rate, log(runs needed / balls left),
               wickets-in-hand x balls-left, pre-match odds
Leverage (LI) of a ball = expected size of the win-probability swing it can cause, using how often each outcome
(dot, 1, 2, 3, 4, 6, wicket, wide/no-ball) happens in that over/wickets situation. LI is scaled so an average ball = 1.
Super-over balls get the 99th-percentile LI.

Pressure score (per ball, separately for the batting side and the bowling side):
    P = 0.6 * z(log LI) + 0.4 * z(log match importance)      (z = standardised over all balls)
Game situation first (60%), match importance second (40%), by design.
Outputs: data/processed/balls_pressure.parquet, outputs/pressure_validation.txt, outputs/*.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data/processed", ROOT / "outputs"
SIT_W, IMP_W = 0.6, 0.4

b = pd.read_parquet(P / "balls.parquet")
ctx = pd.read_parquet(P / "match_team_context.parquet")[["match_id", "team_id", "p0"]]
imp = pd.read_parquet(P / "match_importance.parquet")[["match_id", "team_id", "importance", "reasons"]]
b = b.merge(ctx.rename(columns={"team_id": "batting_team_id"}), on=["match_id", "batting_team_id"], how="left")
b["p0"] = b.p0.fillna(0.5).clip(0.02, 0.98)


def feats(d, inn):
    p0 = np.log(d.p0 / (1 - d.p0))
    if inn == 1:
        left = (d.overs_limit * 6 - d.legal_before).clip(lower=0)
        res = (10 - d.wkts_before).clip(lower=0)
        return pd.DataFrame({"score": d.score_before, "wkts": d.wkts_before, "balls_left": left,
                             "wk_x_bl": res * left / 120, "p0": p0, "sqrt_resources": np.sqrt(res * left)})
    rn, bl = d.runs_needed.clip(lower=1), d.balls_left.clip(lower=1)
    return pd.DataFrame({"runs_needed": rn, "balls_left": bl, "wkts_left": d.wkts_left, "rrr": (rn / bl * 6).clip(upper=36),
                         "log_rn_bl": np.log(rn / bl), "wk_x_bl": d.wkts_left * bl / 120, "p0": p0})


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


main = b[~b.is_super_over]
train = main[main.batting_team_won.isin([0, 1])]
report = []
models = {}
for inn in (1, 2):
    d = train[train.innings == inn]
    X, y, g = feats(d, inn), d.batting_team_won.astype(int), d.match_id
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        oof[te] = model().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    base = np.full(len(d), y.mean())
    report.append(f"Innings {inn} win-prob model (5-fold, split by match): log-loss {log_loss(y, oof):.3f} "
                  f"(no-model {log_loss(y, base):.3f}), Brier {brier_score_loss(y, oof):.3f}, AUC {roc_auc_score(y, oof):.3f}")
    cal = pd.DataFrame({"pred": oof, "y": y.values})
    cal["bin"] = pd.cut(cal.pred, np.linspace(0, 1, 11))
    report.append("  calibration (predicted vs actual win rate):\n" +
                  cal.groupby("bin", observed=True).agg(pred=("pred", "mean"), actual=("y", "mean"), n=("y", "size")).round(3).to_string())
    models[inn] = model().fit(X, y)
    coefs = dict(zip(X.columns, models[inn][-1].coef_[0].round(2)))
    report.append(f"  coefficients (standardised): {coefs}")


def wp(inn, df):
    """Batting-side win probability for a state frame, with hard cricket rules for finished chases."""
    p = models[inn].predict_proba(feats(df, inn))[:, 1]
    if inn == 2:
        p = np.where(df.runs_needed <= 0, 1.0, p)
        p = np.where((df.runs_needed > 0) & ((df.wkts_left <= 0) | (df.balls_left <= 0)), 0.0, p)
    return p


# ----- outcome probabilities by situation (innings x over x wickets bucket) -----
main = main.copy()
main["wb"] = pd.cut(main.wkts_before, [-1, 2, 5, 10], labels=["0-2", "3-5", "6+"]).astype(str)
oc = np.select([~main.legal, main.is_wicket, main.runs >= 6, main.runs == 4, main.runs == 3, main.runs == 2, main.runs == 1],
               ["X", "W", "6", "4", "3", "2", "1"], "0")
main["oc"] = oc
probs = main.groupby(["innings", "overid", "wb"]).oc.value_counts(normalize=True).unstack(fill_value=0)
OUTC = {"0": (0, 0, 1), "1": (1, 0, 1), "2": (2, 0, 1), "3": (3, 0, 1), "4": (4, 0, 1), "6": (6, 0, 1),
        "W": (0, 1, 1), "X": (1, 0, 0)}  # (runs, wicket, legal ball)

li = np.zeros(len(main))
wp_now = np.zeros(len(main))
for inn in (1, 2):
    idx = np.where(main.innings.values == inn)[0]
    s = main.iloc[idx]
    w0 = wp(inn, s)
    wp_now[idx] = w0
    pr = probs.loc[list(zip(s.innings, s.overid, s.wb))].reset_index(drop=True)
    exp_swing = np.zeros(len(s))
    for o, (r, w, lg) in OUTC.items():
        n = s.copy()
        if inn == 1:
            n["score_before"] = s.score_before + r
            n["wkts_before"] = s.wkts_before + w
            n["legal_before"] = s.legal_before + lg
        else:
            n["runs_needed"] = s.runs_needed - r
            n["wkts_left"] = s.wkts_left - w
            n["balls_left"] = s.balls_left - lg
        w1 = wp(inn, n)
        exp_swing += pr[o].values * np.abs(w1 - w0) if o in pr else 0
    li[idx] = exp_swing
# sense check: a run should help the batting side, a wicket or a dot ball should hurt it
for inn in (1, 2):
    s0 = main[main.innings == inn]
    w0 = wp(inn, s0)
    if inn == 1:
        up = wp(inn, s0.assign(score_before=s0.score_before + 1)); wk = wp(inn, s0.assign(wkts_before=s0.wkts_before + 1))
        dot = wp(inn, s0.assign(legal_before=s0.legal_before + 1))
    else:
        up = wp(inn, s0.assign(runs_needed=s0.runs_needed - 1)); wk = wp(inn, s0.assign(wkts_left=s0.wkts_left - 1))
        dot = wp(inn, s0.assign(balls_left=s0.balls_left - 1))
    report.append(f"Sense check innings {inn}: run helps {np.mean(up >= w0):.1%}, wicket hurts {np.mean(wk <= w0):.1%}, "
                  f"dot ball hurts {np.mean(dot <= w0):.1%}")
main["wp_bat"] = wp_now
main["li_raw"] = li
main["LI"] = main.li_raw / main.li_raw.mean()

b = b.merge(main[["match_id", "innings", "innings_ball_num", "wp_bat", "LI"]], on=["match_id", "innings", "innings_ball_num"], how="left")
b.loc[b.is_super_over, "LI"] = main.LI.quantile(0.99)

# ----- pressure score for batting side and bowling side -----
b = b.merge(imp.rename(columns={"team_id": "batting_team_id", "importance": "imp_bat", "reasons": "reasons_bat"}),
            on=["match_id", "batting_team_id"], how="left")
b = b.merge(imp.rename(columns={"team_id": "bowling_team_id", "importance": "imp_bowl", "reasons": "reasons_bowl"}),
            on=["match_id", "bowling_team_id"], how="left")
logli = np.log(b.LI.clip(lower=0.02))
z = lambda x: (x - x.mean()) / x.std()
z_li = z(logli)
allimp = np.log(pd.concat([b.imp_bat, b.imp_bowl]))
zi = lambda x: (np.log(x) - allimp.mean()) / allimp.std()
b["situation_z"] = z_li
b["pressure_bat"] = SIT_W * z_li + IMP_W * zi(b.imp_bat)
b["pressure_bowl"] = SIT_W * z_li + IMP_W * zi(b.imp_bowl)
# percentiles (0-100) make it easy to read: 80+ = top-20% pressure balls
b["pct_bat"] = b.pressure_bat.rank(pct=True) * 100
b["pct_bowl"] = b.pressure_bowl.rank(pct=True) * 100
b.to_parquet(P / "balls_pressure.parquet", index=False)

# ================= validation: do pressure balls look different? =================
b["band"] = pd.cut(b.pct_bat, [0, 50, 80, 100], labels=["Low (bottom 50%)", "Medium (50-80%)", "High (top 20%)"])
lg = b[b.legal]
def stats(g):
    return pd.Series({"balls": len(g), "runs/ball": g.runs.mean(), "dot %": (g.total_runs == 0).mean() * 100,
                      "boundary %": (g.is_four | g.is_six).mean() * 100, "wicket %": g.is_wicket.mean() * 100,
                      "avg LI": g.LI.mean(), "avg importance": g.imp_bat.mean()})
tab = lg.groupby("band", observed=True).apply(stats).round(2)
dec = lg.assign(decile=pd.qcut(lg.pct_bat, 10, labels=range(1, 11))).groupby("decile", observed=True).apply(stats).round(2)
where = pd.crosstab(b.phase, b.band, normalize="columns").round(3) * 100
inn = pd.crosstab(b.innings.clip(upper=3), b.band, normalize="columns").round(3) * 100
comp = pd.crosstab(b.competition, b.band, normalize="index").round(3) * 100
# Like-for-like: same innings, same over, same wickets bucket -> high (top 20%) vs low (bottom 50%) pressure
lg = lg.assign(wb=pd.cut(lg.wkts_before, [-1, 2, 5, 10], labels=["0-2", "3-5", "6+"]).astype(str),
               dot=(lg.total_runs == 0), bnd=(lg.is_four | lg.is_six))
cell = lg[lg.band != "Medium (50-80%)"].groupby(["innings", "overid", "wb", "band"], observed=True).agg(
    n=("runs", "size"), rpb=("runs", "mean"), dot=("dot", "mean"), bnd=("bnd", "mean"), wk=("is_wicket", "mean")).unstack("band")
cell = cell.dropna()
hi, lo = "High (top 20%)", "Low (bottom 50%)"
wts = np.minimum(cell[("n", hi)], cell[("n", lo)])
matched = pd.DataFrame({k: [np.average(cell[(k, lo)], weights=wts), np.average(cell[(k, hi)], weights=wts)]
                        for k in ["rpb", "dot", "bnd", "wk"]}, index=["Low pressure", "High pressure"])
matched.columns = ["runs/ball", "dot rate", "boundary rate", "wicket rate"]
matched.loc["Change"] = (matched.loc["High pressure"] / matched.loc["Low pressure"] - 1)
prof = b[b.legal & ~b.is_super_over].groupby("band", observed=True).agg(
    median_over=("overid", "median"), pct_2nd_inn=("innings", lambda s: (s == 2).mean() * 100),
    median_rrr_chase=("rrr", "median"), median_wkts_left=("wkts_left", "median"),
    pct_close_game=("wp_bat", lambda s: s.between(0.3, 0.7).mean() * 100),
    pct_knockout_or_must_win=("reasons_bat", lambda s: s.str.contains("Final|Qualifier|Eliminator|Semi|do-or-die|decider|Must-win").mean() * 100)).round(1)
# share of variance in pressure explained by each part
sh_sit = np.corrcoef(b.pressure_bat, z_li)[0, 1] ** 2
sh_imp = np.corrcoef(b.pressure_bat, zi(b.imp_bat))[0, 1] ** 2

lines = ["=" * 70, "WIN-PROBABILITY MODEL", *report, "=" * 70,
         f"Pressure score = {SIT_W}*situation + {IMP_W}*match importance (both standardised).",
         f"Share of pressure-score variation explained: situation {sh_sit:.0%}, importance {sh_imp:.0%}",
         "\nLegal balls by pressure band (batting side):\n" + tab.to_string(),
         "\nLIKE-FOR-LIKE (same innings, over and wickets-down; high vs low pressure):\n" + matched.round(3).to_string(),
         "\nWhat each band looks like:\n" + prof.to_string(),
         "\nBy pressure decile (1 = lowest, 10 = highest):\n" + dec.to_string(),
         "\nWhere pressure balls happen - phase (% of each band):\n" + where.to_string(),
         "\nInnings (3 = super over) (% of each band):\n" + inn.to_string(),
         "\nCompetition: % of its balls in each band:\n" + comp.to_string()]
txt = "\n".join(lines)
print(txt)
(OUT / "pressure_validation.txt").write_text(txt, encoding="utf-8")

# ----- charts -----
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, col, lab in zip(ax, ["dot %", "boundary %", "wicket %"], ["Dot-ball %", "Boundary %", "Wicket %"]):
    a.bar(dec.index.astype(str), dec[col], color="#2b6cb0")
    a.set_title(f"{lab} by pressure decile"); a.set_xlabel("Pressure decile (10 = highest)")
plt.tight_layout(); plt.savefig(OUT / "pressure_deciles.png", dpi=130); plt.close()

# Example: closest finish in data - win-probability and leverage through the chase
fin = b[(b.innings == 2) & b.batting_team_won.isin([0, 1])].groupby("match_id").agg(nd=("runs_needed", "last"), bl=("balls_left", "last"))
has2 = b[(b.innings == 2) & b.batting_team_won.isin([0, 1])].match_id.unique()
close = b[b.match_id.isin(has2) & (b.innings == 2)].groupby("match_id").LI.mean().sort_values(ascending=False).index[0]
ex = b[(b.match_id == close) & (b.innings == 2)]
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(ex.innings_ball_num, ex.wp_bat, color="#2b6cb0", label="Chasing side win %")
ax2 = ax1.twinx(); ax2.bar(ex.innings_ball_num, ex.LI, color="#e53e3e", alpha=0.35, label="Leverage")
mt = ex.iloc[0]
ax1.set_title(f"{mt.comp_name}: {mt.home_team} v {mt.away_team} ({mt.match_date}) - chase win % and ball leverage")
ax1.set_xlabel("Ball of chase"); ax1.set_ylabel("Win probability"); ax2.set_ylabel("Leverage (avg ball = 1)")
plt.tight_layout(); plt.savefig(OUT / "example_match_leverage.png", dpi=130); plt.close()
print("\nExample match chart:", mt.comp_name, mt.home_team, "v", mt.away_team, mt.match_date)
