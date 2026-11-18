"""Step 4: Expected outcome of every ball -> value added by batter and bowler -> player tables
(normal level vs pressure level).

Expected outcome (what an average player would do on this ball, in this situation):
   features = innings, over, wickets down, balls left, required rate (chase), pace/spin, competition, season,
              pre-match odds of batting side (opposition strength), and CONDITIONS:
              - venue scoring level in that season (shrunk: venue-season -> venue -> global)
              - "everyone else in this match": run rate and dismissal rate of all other batters in the match
                (batting models) / economy and wicket rate of all other bowlers (bowling models),
                shrunk toward the venue-season level with 120 balls of prior. Rewards good knocks on tough pitches.
   models   = gradient-boosted trees: E[bat runs], P(batter dismissed) on balls faced;
              E[runs charged to bowler], P(bowler's wicket) on all deliveries
Wicket value (runs) = how many runs a team loses on average by losing one more wicket at that stage
   (from a "runs still to come" table built from first innings: balls left x wickets down).
Value per ball:
   batter = (runs - expected runs) - wicket value x (out - expected out)
   bowler = (expected runs conceded - runs conceded) + wicket value x (wicket - expected wicket)
Win impact (WPA, for an optional rating component): change in batting side's win probability over the ball
   (from the step-3 model); bowler gets the opposite sign.
Pressure band per ball: High = top 20% of pressure score (pct >= 80) for that side; Normal = the rest.
Outputs: data/processed/ball_values.parquet, outputs/step4_batters.csv, outputs/step4_bowlers.csv, outputs/step4_summary.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data/processed", ROOT / "outputs"
HIGH = 80  # pressure percentile cut-off for "high pressure"

b = pd.read_parquet(P / "balls_pressure.parquet")
b["season_yr"] = b.season.astype(str).str[:4].astype(int)

# ---------------- runs-still-to-come table and wicket value ----------------
f = b[(b.innings == 1) & (b.overs_limit == 20) & ~b.is_super_over].copy()
f["runs_to_come"] = f.groupby("match_id").total_runs.transform("sum") - f.score_before
re_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=200,
                                         monotonic_cst=[1, -1], random_state=0)
re_model.fit(f[["balls_left", "wkts_before"]], f.runs_to_come)
def RE(bl, w):
    out = re_model.predict(np.c_[bl, np.minimum(w, 9)])
    return np.where(w >= 10, 0.0, out)
bl = b.balls_left.values
w_eff = np.where(b.is_super_over, 8 + b.wkts_before, b.wkts_before)  # super over: 2 wickets = innings
b["wicket_value"] = np.clip(RE(bl, w_eff) - RE(bl, w_eff + 1), 0, None)

# ---------------- expected-outcome models ----------------
b["is_spin"] = (b.pace_or_spin == "Spin").astype(int)
b["comp_code"] = b.competition.map({"IPL": 0, "T20I": 1, "A-team": 2})
b["rrr_f"] = b.rrr.fillna(-1).clip(upper=36)
main = ~b.is_super_over
# --- conditions 1: venue scoring level per season (venue-season -> venue -> global) ---
L = b[b.legal & main]
gm_r, gm_o = L.runs.mean(), L.is_wicket.mean()
ven = L.groupby("venue_id").agg(r=("runs", "sum"), o=("is_wicket", "sum"), n=("runs", "size"))
ven_r, ven_o = (ven.r + 600 * gm_r) / (ven.n + 600), (ven.o + 600 * gm_o) / (ven.n + 600)
vs = L.groupby(["venue_id", "season_yr"]).agg(r=("runs", "sum"), o=("is_wicket", "sum"), n=("runs", "size")).reset_index()
vs["vs_rpb"] = (vs.r + 500 * vs.venue_id.map(ven_r)) / (vs.n + 500)
vs["vs_wpb"] = (vs.o + 500 * vs.venue_id.map(ven_o)) / (vs.n + 500)
b = b.merge(vs[["venue_id", "season_yr", "vs_rpb", "vs_wpb"]], on=["venue_id", "season_yr"], how="left")
b["vs_rpb"] = b.vs_rpb.fillna(gm_r); b["vs_wpb"] = b.vs_wpb.fillna(gm_o)
# --- conditions 2: everyone else in this match, excluding the player being judged ---
# --pitch both (default): all other batters / bowlers from BOTH teams
# --pitch opp           : only the OPPOSITION's batters / bowlers (same pitch, so a strong own attack/line-up does not
#                         make the benchmark harsher). Writes ball_values_opp.parquet (test variant).
PITCH = sys.argv[sys.argv.index("--pitch") + 1] if "--pitch" in sys.argv else "both"
# Opponent quality (DEFAULT ON; --no-quality turns it off): also allow for the quality of the opponent on each ball
# (bowler faced / batter bowled to), measured as that player's runs added vs expected in OTHER matches
# (leave-one-match-out), pulled toward 0 with split-half pull strengths measured on the first-pass values.
# Second pass of the expected models.
QUALITY = "--no-quality" not in sys.argv
K_MATCH = 120
base_r = b.groupby("match_id").vs_rpb.first(); base_w = b.groupby("match_id").vs_wpb.first()
F = b[b.faced & main]
mt = F.groupby("match_id").agg(R=("runs", "sum"), O=("batter_out", "sum"), N=("runs", "size"))
pmb = F.groupby(["match_id", "striker_id"]).agg(r=("runs", "sum"), o=("batter_out", "sum"), n=("runs", "size")).reset_index().join(mt, on="match_id")
if PITCH == "opp":  # "others" = the other team's batters: match totals minus own team's totals
    own_t = F.groupby(["match_id", "striker_id"]).batting_team_id.first()
    tt = F.groupby(["match_id", "batting_team_id"]).agg(R=("runs", "sum"), O=("batter_out", "sum"), N=("runs", "size"))
    k = list(zip(pmb.match_id, own_t.loc[list(zip(pmb.match_id, pmb.striker_id))].values))
    pmb[["r", "o", "n"]] = tt.loc[k].values  # subtract the whole own team instead of just the player
pmb["oth_bat_rpb"] = (pmb.R - pmb.r + K_MATCH * pmb.match_id.map(base_r)) / (pmb.N - pmb.n + K_MATCH)
pmb["oth_bat_opb"] = (pmb.O - pmb.o + K_MATCH * pmb.match_id.map(base_w)) / (pmb.N - pmb.n + K_MATCH)
b = b.merge(pmb[["match_id", "striker_id", "oth_bat_rpb", "oth_bat_opb"]], on=["match_id", "striker_id"], how="left")
mtb = b[main].groupby("match_id").agg(R=("bowler_runs", "sum"), W=("bowler_wicket", "sum"), N=("legal", "sum"))
pbw = b[main].groupby(["match_id", "bowler_id"]).agg(r=("bowler_runs", "sum"), w=("bowler_wicket", "sum"), n=("legal", "sum")).reset_index().join(mtb, on="match_id")
if PITCH == "opp":  # "others" = the opposition's bowlers only
    own_b = b[main].groupby(["match_id", "bowler_id"]).bowling_team_id.first()
    tb = b[main].groupby(["match_id", "bowling_team_id"]).agg(R=("bowler_runs", "sum"), W=("bowler_wicket", "sum"), N=("legal", "sum"))
    k = list(zip(pbw.match_id, own_b.loc[list(zip(pbw.match_id, pbw.bowler_id))].values))
    pbw[["r", "w", "n"]] = tb.loc[k].values
pbw["oth_bowl_rpb"] = (pbw.R - pbw.r + K_MATCH * pbw.match_id.map(base_r)) / (pbw.N - pbw.n + K_MATCH)
pbw["oth_bowl_wpb"] = (pbw.W - pbw.w + K_MATCH * pbw.match_id.map(base_w)) / (pbw.N - pbw.n + K_MATCH)
b = b.merge(pbw[["match_id", "bowler_id", "oth_bowl_rpb", "oth_bowl_wpb"]], on=["match_id", "bowler_id"], how="left")
for c, d in [("oth_bat_rpb", "vs_rpb"), ("oth_bat_opb", "vs_wpb"), ("oth_bowl_rpb", "vs_rpb"), ("oth_bowl_wpb", "vs_wpb")]:
    b[c] = b[c].fillna(b[d])
b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

BASE = ["innings", "overid", "wkts_before", "balls_left", "rrr_f", "is_spin", "comp_code", "season_yr", "p0"]
X_BAT = BASE + ["vs_rpb", "oth_bat_rpb", "oth_bat_opb"]
X_BOWL = BASE + ["vs_rpb", "oth_bowl_rpb", "oth_bowl_wpb"]
hp = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=300, l2_regularization=1.0, random_state=0)

faced = b[b.faced]
reg = lambda: HistGradientBoostingRegressor(**hp)
clf = lambda: HistGradientBoostingClassifier(**hp)
m_runs = reg().fit(faced[X_BAT], faced.runs)
m_out = clf().fit(faced[X_BAT], faced.batter_out.astype(int))
m_bruns = reg().fit(b[X_BOWL], b.bowler_runs)
m_bwk = clf().fit(b[X_BOWL], b.bowler_wicket.astype(int))

b["exp_runs"] = np.where(b.faced, m_runs.predict(b[X_BAT]), np.nan)
b["exp_out"] = np.where(b.faced, m_out.predict_proba(b[X_BAT])[:, 1], np.nan)
b["exp_bruns"] = m_bruns.predict(b[X_BOWL])
b["exp_bwk"] = m_bwk.predict_proba(b[X_BOWL])[:, 1]

if QUALITY:
    def loo_quality(df, pid, val, n, k):
        """Per (player, match): player's average value in all OTHER matches, pulled toward 0 with k balls."""
        t = pd.DataFrame({"pid": df[pid], "match_id": df.match_id, "v": val, "n": n})
        pm = t.groupby(["pid", "match_id"]).sum()
        tot = pm.groupby(level=0).sum()
        oth_v = tot.v.reindex(pm.index.get_level_values(0)).values - pm.v.values
        oth_n = tot.n.reindex(pm.index.get_level_values(0)).values - pm.n.values
        return pd.Series(oth_v / (oth_n + k), index=pm.index)

    def split_half_k(df, pid, val, n, min_balls=600):
        """Pull strength from repeatability, same rule as step 07: each player's matches split into alternate halves
        (players with >= min_balls), r = correlation of the halves, k = n_half (1 - r) / r."""
        t = pd.DataFrame({"pid": df[pid].values, "h": (df.groupby(pid).match_id.rank(method="dense") % 2).values,
                          "s": np.asarray(val, dtype=float), "n": np.asarray(n, dtype=float)})
        g = t.groupby(["pid", "h"]).sum().unstack()
        ok = g[(g[("n", 0)] >= min_balls / 2) & (g[("n", 1)] >= min_balls / 2)]
        r = np.corrcoef(ok[("s", 0)] / ok[("n", 0)], ok[("s", 1)] / ok[("n", 1)])[0, 1]
        nh = np.median(np.r_[ok[("n", 0)], ok[("n", 1)]])
        return nh * (1 - r) / max(r, 0.02)

    allb = b[~b.is_super_over]
    lg = allb.legal.astype(int)
    fb = b[b.faced & ~b.is_super_over]
    one = pd.Series(1, index=fb.index)
    parts = {"bowl economy": (allb, "bowler_id", allb.exp_bruns - allb.bowler_runs, lg),
             "bowl wickets": (allb, "bowler_id", allb.wicket_value * (allb.bowler_wicket.astype(float) - allb.exp_bwk), lg),
             "bat scoring": (fb, "striker_id", fb.runs - fb.exp_runs, one),
             "bat dismissals": (fb, "striker_id", fb.wicket_value * (fb.exp_out - fb.batter_out.astype(float)), one)}
    # pull strengths are measured on this data (first-pass values), never typed in, so they stay right if the data change
    K_Q = {name: split_half_k(*p) for name, p in parts.items()}
    print("quality pull strengths k (balls): " + ", ".join(f"{name} {k:.0f}" for name, k in K_Q.items()))
    q_be = loo_quality(*parts["bowl economy"], K_Q["bowl economy"])
    q_bw = loo_quality(*parts["bowl wickets"], K_Q["bowl wickets"])
    q_ss = loo_quality(*parts["bat scoring"], K_Q["bat scoring"])
    q_so = loo_quality(*parts["bat dismissals"], K_Q["bat dismissals"])
    kb = pd.MultiIndex.from_arrays([b.bowler_id, b.match_id]); ks = pd.MultiIndex.from_arrays([b.striker_id, b.match_id])
    b["opp_bowler_q_econ"] = q_be.reindex(kb).fillna(0).values
    b["opp_bowler_q_wkt"] = q_bw.reindex(kb).fillna(0).values
    b["opp_batter_q_score"] = q_ss.reindex(ks).fillna(0).values
    b["opp_batter_q_out"] = q_so.reindex(ks).fillna(0).values
    X_BAT = X_BAT + ["opp_bowler_q_econ", "opp_bowler_q_wkt"]
    X_BOWL = X_BOWL + ["opp_batter_q_score", "opp_batter_q_out"]
    faced = b[b.faced]
    m_runs = reg().fit(faced[X_BAT], faced.runs)
    m_out = clf().fit(faced[X_BAT], faced.batter_out.astype(int))
    m_bruns = reg().fit(b[X_BOWL], b.bowler_runs)
    m_bwk = clf().fit(b[X_BOWL], b.bowler_wicket.astype(int))
    b["exp_runs"] = np.where(b.faced, m_runs.predict(b[X_BAT]), np.nan)
    b["exp_out"] = np.where(b.faced, m_out.predict_proba(b[X_BAT])[:, 1], np.nan)
    b["exp_bruns"] = m_bruns.predict(b[X_BOWL])
    b["exp_bwk"] = m_bwk.predict_proba(b[X_BOWL])[:, 1]
    print("quality pass done; spread of opponent quality (runs/ball sd): bowler econ "
          f"{b.opp_bowler_q_econ.std():.3f}, batter scoring {b.opp_batter_q_score.std():.3f}")

# ---------------- win impact (WPA) per ball ----------------
g = b.groupby(["match_id", "innings"])
wp_after = g.wp_bat.shift(-1)
last = wp_after.isna() & b.wp_bat.notna()
first2 = b[b.innings == 2].groupby("match_id").wp_bat.first()          # chasing side's WP before ball 1 of chase
end1 = last & (b.innings == 1)
wp_after = wp_after.where(~end1, 1 - b.match_id.map(first2))
end2 = last & (b.innings == 2)
final = np.where(b.tied, 0.5, b.batting_team_won)
wp_after = wp_after.where(~end2, pd.Series(final, index=b.index))
b["wpa_bat"] = (wp_after - b.wp_bat).fillna(0.0)                       # super overs / no results -> 0
b.loc[b.is_super_over | b.no_result, "wpa_bat"] = 0.0
b["wpa_bowl"] = -b.wpa_bat

b["bat_value"] = np.where(b.faced, (b.runs - b.exp_runs) - b.wicket_value * (b.batter_out - b.exp_out), np.nan)
b["bowl_value"] = (b.exp_bruns - b.bowler_runs) + b.wicket_value * (b.bowler_wicket - b.exp_bwk)
b["bat_high"] = b.pct_bat >= HIGH
b["bowl_high"] = b.pct_bowl >= HIGH

keep = ["match_id", "innings", "innings_ball_num", "match_date", "competition", "comp_name", "season", "phase", "overid",
        "striker_id", "striker_name", "bowler_id", "bowler_name", "batting_team_id", "bowling_team_id",
        "legal", "faced", "runs", "bowler_runs", "batter_out", "bowler_wicket", "is_four", "is_six",
        "exp_runs", "exp_out", "exp_bruns", "exp_bwk", "wicket_value", "bat_value", "bowl_value",
        "LI", "imp_bat", "imp_bowl", "pressure_bat", "pressure_bowl", "pct_bat", "pct_bowl", "bat_high", "bowl_high",
        "is_super_over", "wp_bat", "wpa_bat", "wpa_bowl", "vs_rpb", "oth_bat_rpb", "oth_bowl_rpb"]
b[keep].to_parquet(P / ("ball_values_opp.parquet" if PITCH == "opp" else "ball_values.parquet"), index=False)
if PITCH == "opp":
    print("opposition-only pitch variant written"); sys.exit(0)

# sanity: values should average ~0 over everyone
chk = f"Average value per ball: batting {b.bat_value.mean():+.4f}, bowling {b.bowl_value.mean():+.4f} (should be ~0)\n"
chk += f"Average wicket value: {b.wicket_value.mean():.1f} runs (powerplay {b[b.phase=='Powerplay'].wicket_value.mean():.1f}, " \
       f"death {b[b.phase=='Death'].wicket_value.mean():.1f})\n"

# ---------------- player tables ----------------
def bat_band(g):
    n = len(g)
    outs, xo = g.batter_out.sum(), g.exp_out.sum()
    return pd.Series({
        "balls": n, "runs": g.runs.sum(), "SR": 100 * g.runs.sum() / n, "xSR": 100 * g.exp_runs.sum() / n,
        "outs": outs, "x_outs": xo, "balls_per_out": n / max(outs, 1), "x_balls_per_out": n / max(xo, 1e-9),
        "value": g.bat_value.sum(), "value_per100": 100 * g.bat_value.mean(),
        "innings": g.match_id.nunique()})

def bowl_band(g):
    lg = g.legal.sum()
    return pd.Series({
        "balls": lg, "runs": g.bowler_runs.sum(), "econ": 6 * g.bowler_runs.sum() / max(lg, 1),
        "x_econ": 6 * g.exp_bruns.sum() / max(lg, 1), "wkts": g.bowler_wicket.sum(), "x_wkts": g.exp_bwk.sum(),
        "value": g.bowl_value.sum(), "value_per100": 100 * g.bowl_value.sum() / max(lg, 1),
        "innings": g.match_id.nunique()})

def consistency(df, id_col, val, high_col, min_balls):
    """Per match: value in high-pressure balls; share of 'pressure innings' (>= min_balls high balls) with positive value."""
    h = df[df[high_col]].groupby([id_col, "match_id"]).agg(v=(val, "sum"), n=(val, "size"))
    h = h[h.n >= min_balls]
    return h.groupby(id_col).agg(pressure_inns=("v", "size"), pct_positive=("v", lambda s: 100 * (s > 0).mean()),
                                 median_inn_value=("v", "median"))

def build(df, id_col, name_col, band_fn, val, high_col, min_inn_balls):
    names = df.groupby(id_col)[name_col].agg(lambda s: s.mode().iloc[0])
    allb = df.groupby(id_col).apply(band_fn).add_prefix("all_")
    norm = df[~df[high_col]].groupby(id_col).apply(band_fn).add_prefix("norm_")
    high = df[df[high_col]].groupby(id_col).apply(band_fn).add_prefix("pres_")
    t = pd.concat([names.rename("player"), allb, norm, high], axis=1)
    t = t.join(consistency(df, id_col, val, high_col, min_inn_balls))
    comp = df[df[high_col]].groupby([id_col, "competition"]).size().unstack(fill_value=0).add_prefix("pres_balls_")
    t = t.join(comp).fillna({c: 0 for c in comp.columns})
    t["gap_per100"] = t.pres_value_per100 - t.norm_value_per100
    return t

bat = build(b[b.faced], "striker_id", "striker_name", bat_band, "bat_value", "bat_high", 5)
bowl = build(b, "bowler_id", "bowler_name", bowl_band, "bowl_value", "bowl_high", 6)
bat.sort_values("pres_balls", ascending=False).round(2).to_csv(OUT / "step4_batters.csv")
bowl.sort_values("pres_balls", ascending=False).round(2).to_csv(OUT / "step4_bowlers.csv")

# ---------------- summary ----------------
lines = [chk]
for lab, t in [("BATTERS", bat), ("BOWLERS", bowl)]:
    pb = t.pres_balls.fillna(0)
    lines.append(f"{lab}: players {len(t)} | high-pressure balls per player: " +
                 ", ".join(f">= {x}: {int((pb >= x).sum())}" for x in [50, 75, 100, 150, 200, 250, 300, 400]))
    lines.append(f"  correlation normal-level vs pressure-level value (players with >=150 pressure balls): "
                 f"{t[pb >= 150][['norm_value_per100', 'pres_value_per100']].corr().iloc[0, 1]:.2f}")
    top = t[pb >= 150].sort_values("pres_value_per100", ascending=False).head(15)
    cols = (["player", "all_balls", "pres_balls", "norm_SR", "pres_SR", "pres_xSR", "pres_outs", "pres_x_outs",
             "norm_value_per100", "pres_value_per100", "gap_per100", "pressure_inns", "pct_positive"] if lab == "BATTERS" else
            ["player", "all_balls", "pres_balls", "norm_econ", "pres_econ", "pres_x_econ", "pres_wkts", "pres_x_wkts",
             "norm_value_per100", "pres_value_per100", "gap_per100", "pressure_inns", "pct_positive"])
    lines.append(f"  Raw leaders by pressure value per 100 balls (>=150 pressure balls, NO shrinkage yet):\n" +
                 top[cols].round(1).to_string(index=False))

# duplicate-name check
for lab, df, idc, nc in [("batters", b[b.faced], "striker_id", "striker_name"), ("bowlers", b, "bowler_id", "bowler_name")]:
    d = df.groupby(nc)[idc].nunique()
    for nm in d[d > 1].index:
        info = df[df[nc] == nm].groupby(idc).agg(balls=(idc, "size"), first=("match_date", "min"), last=("match_date", "max"),
                                                  comps=("competition", lambda s: ",".join(sorted(set(s)))))
        lines.append(f"Same name, different IDs ({lab}) - {nm}:\n{info.to_string()}")
txt = "\n".join(lines)
print(txt)
(OUT / "step4_summary.txt").write_text(txt, encoding="utf-8")
