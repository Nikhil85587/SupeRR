"""Step 5: Pressure ratings, rankings at 3 minimum-sample levels, and robustness checks.

For each player (batters: balls faced; bowlers: legal balls) - all values are runs above expected per 100 balls:
 1. Normal level  = value on normal balls (pressure pct < cut), pulled toward the league average
                    (empirical Bayes: weight = n / (n + k_norm)).
 2. Pressure level = value on high-pressure balls, pulled toward (player's normal level + league-wide pressure shift)
                    with weight n / (n + k_pres).
    Pull strength k (in balls) for each part comes from split-half repeatability: k = n_half (1 - r) / r
    (players with >= 600 balls split by alternate matches). Pressure k = min(method-of-moments estimate, normal k).
    Inside the pressure band, each ball is weighted by sqrt(match importance / average pressure-ball importance),
    so knockout / must-win / rivalry balls count more (moderate).
 3. Consistency   = share of pressure innings (>=5 pressure balls faced / >=6 bowled) beating expectation,
                    pulled toward the league share with 5 innings of prior.
 4. Rise          = pressure level - normal level (both after the pull).
 3b. Consistency (default "robust") = average of z(share of good pressure innings), z(typical/median pressure innings),
     -z(bad-day rate; bad day = below the league's bottom 20% of pressure innings), each pulled with 5 innings of prior.
 Final score = 0.625 z(pressure level) + 0.225 z(consistency) + 0.15 z(rise)   (z within the eligible pool)
   (brief asks for players who "deliver most consistently" and to compare with "usual performance levels")
Optional (test only): --win-share S blends win impact (WPA) into the pressure level:
 pressure part of score = (1-S) z(runs-based pressure level) + S z(win-impact pressure level). Outputs get suffix _winXX.
Eligibility: minimum high-pressure balls = 100 / 150 / 250 (3 separate runs).
Robustness: bootstrap over matches (90% rank range), high-pressure cut at top 15% / 25%, IPL-only and T20I-only ranks.
Outputs: outputs/step5_<bat|bowl>_min<N>.csv, outputs/step5_notes.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P, OUT = ROOT / "data/processed", ROOT / "outputs"
THRESHOLDS = [100, 150, 250]
WEIGHTS = dict(pres=0.625, cons=0.225, rise=0.15)  # the brief asks for consistency -> 22.5%
N_BOOT = 300
WIN_SHARE = float(sys.argv[sys.argv.index("--win-share") + 1]) if "--win-share" in sys.argv else 0.0
CONS_MODE = sys.argv[sys.argv.index("--cons") + 1] if "--cons" in sys.argv else "robust"     # robust (default) | simple
VALUES = sys.argv[sys.argv.index("--values") + 1] if "--values" in sys.argv else "both"      # both | opp | quality
RISE_MODE = sys.argv[sys.argv.index("--rise") + 1] if "--rise" in sys.argv else "both"       # both | positive
SUFFIX = (f"_win{int(round(WIN_SHARE * 100))}" if WIN_SHARE > 0 else "") + \
         ("_consS" if CONS_MODE == "simple" else "") + ("_riseP" if RISE_MODE == "positive" else "") +          {"both": "", "opp": "_pitchOpp", "quality": "_quality"}[VALUES]
BAD_Q = 0.20       # a "bad day" = pressure innings worse than the league's bottom 20% of pressure innings
BAD_THR = {}
IMP_POWER = 0.5  # knockout weighting strength inside the pressure band
# Recency: OFF by default since 2026-11-17 (organiser email: the task is NOT an auction pick, so every season 2022-26 counts
# equally). `--recency` restores the old season weights (2022 0.4 ... 2026 1.0) as a check; files get suffix _rec.
RECENCY = {2022: 0.4, 2023: 0.55, 2024: 0.7, 2025: 0.85, 2026: 1.0} if "--recency" in sys.argv else {}
if RECENCY:
    SUFFIX += "_rec"
rng = np.random.default_rng(42)

v = pd.read_parquet(P / {"both": "ball_values.parquet", "opp": "ball_values_opp.parquet",
                          "quality": "ball_values_quality.parquet"}[VALUES])
# Value split into two parts that behave differently (economy/scoring repeats well; wickets/dismissals are noisier)
v["bat_score_part"] = v.runs - v.exp_runs
v["bat_out_part"] = -v.wicket_value * (v.batter_out.astype(float) - v.exp_out)
v["bowl_econ_part"] = v.exp_bruns - v.bowler_runs
v["bowl_wkt_part"] = v.wicket_value * (v.bowler_wicket.astype(float) - v.exp_bwk)
PARTS = {"bat": ["bat_score_part", "bat_out_part", "wpa_bat"], "bowl": ["bowl_econ_part", "bowl_wkt_part", "wpa_bowl"]}
N_RUN_PARTS = 2  # parts 0-1 are in runs and are added up; part 2 is win impact (kept separate)
teams = pd.read_csv(ROOT / "Updated Data" / "teams.csv").drop_duplicates("team_id").set_index("team_id").team_name

ROLE = {
    "bat": dict(id="striker_id", name="striker_name", val="bat_value", pct="pct_bat", team="batting_team_id", imp="imp_bat",
                rows=v[v.faced], n_col=None, min_inn=5),
    "bowl": dict(id="bowler_id", name="bowler_name", val="bowl_value", pct="pct_bowl", team="bowling_team_id", imp="imp_bowl",
                 rows=v, n_col="legal", min_inn=6),
}


def player_match(role, cut, comp=None):
    """Per player-match sums: pressure value/balls, normal value/balls, positive pressure innings flag."""
    R = ROLE[role]
    d = R["rows"] if comp is None else R["rows"][R["rows"].competition == comp]
    hi = d[R["pct"]] >= cut
    n = d[R["n_col"]].astype(int) if R["n_col"] else pd.Series(1, index=d.index)
    # big-match weight inside the pressure band (normal balls weight 1)
    imp_hi = R["rows"].loc[R["rows"][R["pct"]] >= cut, R["imp"]].mean()
    rec = d.season.astype(str).str[:4].astype(int).map(RECENCY).fillna(1.0).values if RECENCY else np.ones(len(d))
    w = np.where(hi, (d[R["imp"]] / imp_hi) ** IMP_POWER, 1.0) * rec
    t = pd.DataFrame({"pid": d[R["id"]], "match_id": d.match_id,
                      "hv": np.where(hi, w * d[R["val"]], 0.0), "hn": np.where(hi, w * n, 0.0),
                      "hn_raw": np.where(hi, n, 0), "nv": np.where(~hi, rec * d[R["val"]], 0.0),
                      "nn": np.where(~hi, rec * n, 0.0), "rec": rec})
    for i, part in enumerate(PARTS[role]):
        t[f"hv{i}"] = np.where(hi, w * d[part], 0.0)
        t[f"nv{i}"] = np.where(~hi, rec * d[part], 0.0)
    match_rec = t.groupby("match_id").rec.first()
    pm = t.drop(columns="rec").groupby(["pid", "match_id"], as_index=False).sum()
    flag = (pm.hn_raw >= R["min_inn"])
    pm["p_inn_raw"] = flag.astype(int)
    pm["p_inn"] = flag * pm.match_id.map(match_rec)          # innings weighted by recency
    pm["p_pos"] = (flag & (pm.hv > 0)) * pm.match_id.map(match_rec)
    return pm


def noise_var(role, cut=None):
    """Per-ball noise variance of each value part (pooled within-player variance)."""
    R = ROLE[role]
    d = R["rows"]
    return [float((d[p] - d[p].groupby(d[R["id"]]).transform("mean")).var()) for p in PARTS[role]]


def split_half_k(role, part, min_balls=600, high_only=False):
    """Pull strength from repeatability: split each player's matches into alternate halves, correlate the halves.
    r = n_half / (n_half + k)  ->  k = n_half (1 - r) / r."""
    R = ROLE[role]; d = R["rows"]
    if high_only:
        d = d[d[R["pct"]] >= 80]
    n = d[R["n_col"]].astype(int) if R["n_col"] else pd.Series(1, index=d.index)
    h = d.groupby(R["id"]).match_id.rank(method="dense") % 2
    g = pd.DataFrame({"pid": d[R["id"]], "h": h, "s": d[part], "n": n}).groupby(["pid", "h"]).sum().unstack()
    ok = g[(g[("n", 0)] >= min_balls / 2) & (g[("n", 1)] >= min_balls / 2)]
    r = np.corrcoef(ok[("s", 0)] / ok[("n", 0)], ok[("s", 1)] / ok[("n", 1)])[0, 1]
    nh = np.median(np.r_[ok[("n", 0)], ok[("n", 1)]])
    return nh * (1 - r) / max(r, 0.02), r


def k_from_moments(mean, n, sigma2, min_n=100):
    """Method of moments: true spread tau^2 = var(player means) - avg(sigma^2/n). k = sigma^2 / tau^2."""
    ok = n >= min_n
    tau2 = np.var(mean[ok]) - np.mean(sigma2 / n[ok])
    tau2 = max(tau2, 1e-6)
    return sigma2 / tau2, tau2


SUM_COLS = ["hv", "hn", "hn_raw", "nv", "nn", "p_inn", "p_inn_raw", "p_pos", "hv0", "nv0", "hv1", "nv1", "hv2", "nv2"]


def rate(pm, sigma2, k_norm=None, k_pres=None, role=None):
    """sigma2, k_norm, k_pres are lists (one per value part). Each part is pulled separately; run parts are added."""
    g = pm.groupby("pid")[SUM_COLS].sum()
    g = g[(g.hn > 0) & (g.nn > 0)]
    fit = k_norm is None
    if fit:
        k_norm, k_pres = [split_half_k(role, p)[0] for p in PARTS[role]], []
    norm_tot = pres_tot = 0
    shift_tot = 0.0
    for i in range(len(k_norm)):
        norm_raw, pres_raw = g[f"nv{i}"] / g.nn, g[f"hv{i}"] / g.hn
        mu = g[f"nv{i}"].sum() / g.nn.sum()
        norm = mu + (norm_raw - mu) * g.nn / (g.nn + k_norm[i])
        shift = g[f"hv{i}"].sum() / g.hn.sum() - mu  # league-wide average change under pressure
        prior = norm + shift
        if fit:
            dev = (pres_raw - prior).values
            # rule: pressure evidence is never trusted less than the same part at normal level
            if i < N_RUN_PARTS:
                k_pres.append(min(k_from_moments(dev, g.hn.values, sigma2[i])[0], k_norm[i]))
            else:  # win impact: pressure-ball swings are much bigger, so measure repeatability on pressure balls
                k_pres.append(split_half_k(role, PARTS[role][i], min_balls=150, high_only=True)[0])
        pres = prior + (pres_raw - prior) * g.hn / (g.hn + k_pres[i])
        if i < N_RUN_PARTS:
            norm_tot = norm_tot + norm; pres_tot = pres_tot + pres; shift_tot += shift
        else:
            win_norm, win_pres = norm, pres
    lg_pos = g.p_pos.sum() / max(g.p_inn.sum(), 1)
    cons = (g.p_pos + 5 * lg_pos) / (g.p_inn + 5)
    # robust consistency inputs: typical (median) pressure innings and rate of bad days, each pulled with 5 innings
    inn = pm[pm.p_inn > 0][["pid", "hv", "p_inn"]].copy()
    inn["val"] = inn.hv / inn.p_inn            # value of that pressure innings (bootstrap weights cancel out)
    if role not in BAD_THR:
        BAD_THR[role] = inn.val.quantile(BAD_Q)
    inn["bad"] = (inn.val < BAD_THR[role]) * inn.p_inn
    lg_med = inn.val.median(); lg_bad = inn.bad.sum() / inn.p_inn.sum()
    inn = inn.sort_values(["pid", "val"])
    inn["cw"] = inn.groupby("pid").p_inn.cumsum(); inn["tw"] = inn.groupby("pid").p_inn.transform("sum")
    med = inn[inn.cw >= inn.tw / 2].groupby("pid").val.first()
    nin = inn.groupby("pid").p_inn.sum(); nbad = inn.groupby("pid").bad.sum()
    med_s = ((med * nin + 5 * lg_med) / (nin + 5)).reindex(g.index).fillna(lg_med)
    bad_s = ((nbad + 5 * lg_bad) / (nin + 5)).reindex(g.index).fillna(lg_bad)
    out = pd.DataFrame({"pres_balls": g.hn_raw, "pres_balls_weighted": g.hn, "norm_balls": g.nn, "pressure_innings": g.p_inn_raw,
                        "win_normal": 100 * win_norm, "win_pressure": 100 * win_pres,
                        "raw_pres_per100": 100 * g.hv / g.hn, "raw_norm_per100": 100 * g.nv / g.nn,
                        "normal_level": 100 * norm_tot, "pressure_level": 100 * pres_tot,
                        "consistency_pct": 100 * cons, "median_pressure_innings": med_s, "bad_day_pct": 100 * bad_s,
                        "rise": 100 * (pres_tot - norm_tot)})
    return out, k_norm, k_pres, shift_tot


def final_score(e):
    z = lambda s: (s - s.mean()) / s.std()
    pres = (1 - WIN_SHARE) * z(e.pressure_level) + WIN_SHARE * z(e.win_pressure) if WIN_SHARE > 0 else z(e.pressure_level)
    if CONS_MODE == "robust":   # share of good innings + typical innings - bad days
        cons = (z(e.consistency_pct) + z(e.median_pressure_innings) - z(e.bad_day_pct)) / 3
    else:
        cons = e.consistency_pct
    rise = e.rise.clip(lower=0) if RISE_MODE == "positive" else e.rise
    return WEIGHTS["pres"] * pres + WEIGHTS["cons"] * z(cons) + WEIGHTS["rise"] * z(rise)


def score(t, min_balls):
    e = t[t.pres_balls >= min_balls].copy()
    e["final_score"] = final_score(e)
    e["rank"] = e.final_score.rank(ascending=False, method="first").astype(int)
    return e.sort_values("rank")


def describe(role):
    R = ROLE[role]
    d = R["rows"]
    name = d.groupby(R["id"])[R["name"]].agg(lambda s: s.mode().iloc[0])
    tm = d.assign(team=d[R["team"]].map(teams))
    intl = tm[tm.competition == "T20I"].groupby(R["id"]).team.agg(lambda s: s.mode().iloc[0])
    ipl = tm[tm.competition == "IPL"].groupby(R["id"]).team.agg(lambda s: "/".join(sorted(set(s))))
    extra = pd.DataFrame({"player": name, "country": intl, "ipl_teams": ipl})
    if role == "bat":
        hi = d[d.pct_bat >= 80].groupby(R["id"]).agg(r=("runs", "sum"), xr=("exp_runs", "sum"), n=("runs", "size"),
                                                    o=("batter_out", "sum"), xo=("exp_out", "sum"))
        extra["pres_SR"] = 100 * hi.r / hi.n; extra["pres_xSR"] = 100 * hi.xr / hi.n
        extra["pres_outs"] = hi.o; extra["pres_x_outs"] = hi.xo.round(1)
    else:
        hi = d[d.pct_bowl >= 80].groupby(R["id"]).agg(r=("bowler_runs", "sum"), xr=("exp_bruns", "sum"), n=("legal", "sum"),
                                                     w=("bowler_wicket", "sum"), xw=("exp_bwk", "sum"))
        extra["pres_econ"] = 6 * hi.r / hi.n; extra["pres_x_econ"] = 6 * hi.xr / hi.n
        extra["pres_wkts"] = hi.w; extra["pres_x_wkts"] = hi.xw.round(1)
    return extra


notes = []
results = {}
for role in ["bat", "bowl"]:
    sigma2 = noise_var(role, 80)
    pm = player_match(role, 80)
    base, k_norm, k_pres, shift = rate(pm, sigma2, role=role)
    for i, part in enumerate(PARTS[role]):
        notes.append(f"[{role}] {part}: split-half r {split_half_k(role, part)[1]:.2f} | pull strength: normal level k={k_norm[i]:.0f} balls, "
                     f"pressure level k={k_pres[i]:.0f} balls")
    notes.append(f"[{role}] league-wide change under pressure {100*shift:+.2f} runs/100 balls")
    # repeatability of pressure form itself (quoted in the report): total value on pressure balls only, >= 75 per half
    notes.append(f"[{role}] pressure balls only, total value: split-half r "
                 f"{split_half_k(role, ROLE[role]['val'], min_balls=150, high_only=True)[1]:.2f}")
    extra = describe(role)
    # robustness inputs computed once
    alt = {c: rate(player_match(role, c), sigma2, k_norm, k_pres, role=role)[0] for c in (75, 85)}
    comp_rank = {}
    for comp in ["IPL", "T20I"]:
        t, *_ = rate(player_match(role, 80, comp), sigma2, k_norm, k_pres, role=role)
        comp_rank[comp] = score(t, 60)["rank"]
    # bootstrap over matches
    matches = pm.match_id.unique()
    boots = []
    for _ in range(N_BOOT):
        w = pd.Series(rng.multinomial(len(matches), np.ones(len(matches)) / len(matches)), index=matches)
        pmb = pm.copy()
        ww = pmb.match_id.map(w).values
        pmb[SUM_COLS] = pmb[SUM_COLS].mul(ww, axis=0)
        boots.append(rate(pmb, sigma2, k_norm, k_pres, role=role)[0])

    for th in THRESHOLDS:
        e = score(base, th)
        ranks = []
        for bt in boots:
            bt = bt.reindex(e.index).dropna(subset=["pressure_level"])
            ranks.append(final_score(bt).rank(ascending=False))
        rk = pd.concat(ranks, axis=1)
        e["rank_5pct"] = rk.quantile(0.05, axis=1).round().astype("Int64")
        e["rank_95pct"] = rk.quantile(0.95, axis=1).round().astype("Int64")
        e["pct_boot_top10"] = (rk <= 10).mean(axis=1) * 100
        for c, lab in [(85, "rank_top15pct_cut"), (75, "rank_top25pct_cut")]:
            ae = alt[c].reindex(e.index)
            e[lab] = final_score(ae).rank(ascending=False).astype("Int64")
        e["rank_IPL_only"] = comp_rank["IPL"].reindex(e.index).astype("Int64")
        e["rank_T20I_only"] = comp_rank["T20I"].reindex(e.index).astype("Int64")
        e = extra.join(e, how="inner").sort_values("rank")
        cols = ["rank", "player", "country", "ipl_teams", "pres_balls", "pressure_innings"] + \
               (["pres_SR", "pres_xSR", "pres_outs", "pres_x_outs"] if role == "bat" else ["pres_econ", "pres_x_econ", "pres_wkts", "pres_x_wkts"]) + \
               ["normal_level", "pressure_level", "win_pressure", "rise", "consistency_pct", "median_pressure_innings", "bad_day_pct", "final_score", "rank_5pct", "rank_95pct", "pct_boot_top10",
                "rank_top15pct_cut", "rank_top25pct_cut", "rank_IPL_only", "rank_T20I_only", "raw_pres_per100"]
        e = e[cols]
        e.round(2).to_csv(OUT / f"step5_{role}_min{th}{SUFFIX}.csv", index=False)
        results[(role, th)] = e
        notes.append(f"[{role} min {th}] eligible players: {len(e)}")

pd.set_option("display.width", 250)
show = {"bat": ["rank", "player", "country", "pres_balls", "pressure_innings", "pres_SR", "pres_xSR", "pres_outs", "pres_x_outs",
                "normal_level", "pressure_level", "consistency_pct", "rank_5pct", "rank_95pct", "rank_IPL_only", "rank_T20I_only"],
        "bowl": ["rank", "player", "country", "pres_balls", "pressure_innings", "pres_econ", "pres_x_econ", "pres_wkts", "pres_x_wkts",
                 "normal_level", "pressure_level", "consistency_pct", "rank_5pct", "rank_95pct", "rank_IPL_only", "rank_T20I_only"]}
for (role, th), e in results.items():
    notes.append(f"\n===== {'BATTERS' if role == 'bat' else 'BOWLERS'} | min {th} pressure balls | top 25 =====\n" +
                 e.head(25)[show[role]].round(1).to_string(index=False))
txt = "\n".join(notes)
print(txt)
(OUT / f"step5_notes{SUFFIX}.txt").write_text(txt, encoding="utf-8")
