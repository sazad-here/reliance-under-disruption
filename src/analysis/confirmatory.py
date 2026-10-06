"""Pre-registered confirmatory models, OSF section 10.

Coding (so every coefficient is a difference, averaged over the other factor):
    rel_c   = +0.5 for 80 % ("high"), -0.5 for 60 % ("low")  -> coef = high - low
    place_c = +0.5 propagating, -0.5 absorbed                -> coef = prop - abs
    block   = categorical 0/1/2 (treatment-coded, block 0 reference), as
              registered ("Trial enters as block rather than a linear term")

Models (random intercept per participant; "random slopes on trial where they
converge" is attempted for the trial-level linear models as a random slope on
centred trial index, and dropped -- recorded in the JSON -- if the fit does
not converge or the slope variance is on the boundary):

  H1   WoA ~ rel_c + ai_correct + C(block)                  test rel_c > 0
  H2   trust ~ rel_c * C(block)   (block-level, 3 per person)
                                  test: joint Wald (2 df) on the two
                                  interaction terms; estimate reported is the
                                  block-3 minus block-1 divergence (rel_c:block2)
  H3   WoA ~ place_c * rel_c   on post-error trials         test place_c < 0
       (post-error = the trial after an AI error; placement = that error's
       placement; see measures.py)
       CAUTION (found on simulated data): propagating weeks cluster (7-8-9,
       14-15-16), so a trial after a propagating error is itself an AI error
       far more often than a trial after an absorbed error (60 %: ~57 % vs
       ~27 %). The registered formula does not adjust for this; the JSON
       carries a sensitivity model adding ai_correct, and the shares.
  H4a  |Q_final - Q*| ~ rel_c + C(block)   all main trials   test rel_c < 0
  H4b  |Q_final - Q*| ~ rel_c * place_c    post-error trials test rel_c:place_c < 0
       (the 60 %-minus-80 % gap is larger after propagating errors)
  AR   logit P(AR = 1) ~ rel_c * C(block) + u_i   (the registered binary
       model; AR from measures.py, defined on Schemmer CA/IA trials)
       test: rel_c (two-sided; the registration names no direction)

GLMM method: statsmodels has no frequentist binomial GLMM (only the
variational-Bayes BinomialBayesMixedGLM). The AR model is therefore fitted by
MAXIMUM LIKELIHOOD with the random intercept integrated out by Gauss-Hermite
quadrature (30 nodes; the same estimator as lme4::glmer with nAGQ > 1, but
non-adaptive), Wald SEs from the numerical Hessian. As robustness checks the
results JSON also carries a GEE (exchangeable working correlation, robust
SEs) and the statsmodels BinomialBayesMixedGLM (VB) estimate.

Holm: applied across the family {H1, H2, H3, H4a, H4b, AR} -- every
confirmatory test on a primary outcome or a registered hypothesis. The
registration says "across the primary outcomes"; including H2 (trust is listed
as secondary) makes the family larger and the correction more conservative.

Supported = Holm-adjusted p < .05 AND the estimate in the predicted direction
(H2 and AR are non-directional).

Effect sizes: raw coefficients with 95 % Wald CIs; for the linear models also
a standardised d = b / sqrt(var_u + var_e) (CI = CI of b over the same
denominator, treating the variance components as fixed); for AR the odds
ratio.

Operational outcome (secondary, not in the Holm family): realised cost is
aggregated to block level (the registered alternative to an AR(1) residual,
which statsmodels MixedLM does not provide): block cost ~ rel_c + C(block).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_hess

ALPHA = 0.05
FAMILY = ("H1", "H2", "H3", "H4a", "H4b", "AR")


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rel_c"] = np.where(d["condition"] == "high", 0.5, -0.5)
    if "post_error_placement" in d:
        d["place_c"] = d["post_error_placement"].map({"propagating": 0.5, "absorbed": -0.5})
    d["block"] = d["block"].astype(int)
    if "trial_index" in d:
        d["trial_c"] = (d["trial_index"] - 14.5) / 10.0
    return d


# ------------------------------------------------------------ linear mixed
def _fit_mixed(formula, d, try_slope=True):
    """MixedLM, random intercept (+ random slope on trial_c if it converges)."""
    attempts = [("~trial_c", "random intercept + slope on trial")] if try_slope else []
    attempts.append((None, "random intercept"))
    last_err = None
    # REML. Optimisers are tried in a fixed order and the first CLEAN fit (no
    # warning, finite SEs) is used; otherwise the converged fit with the
    # highest REML log-likelihood is kept. Reason: statsmodels' L-BFGS can
    # report convergence at a spurious zero random-intercept variance (singular
    # covariance, absurd between-subject SEs) where Powell / Nelder-Mead reach
    # the real optimum. Fits flagged singular, or with non-finite SEs, are only
    # used when nothing else converged; a random slope is kept only if some fit
    # has no convergence/boundary/singular warning at all.
    for re_formula, label in attempts:
        fits = []
        # The slope is judged by whether the default optimiser (L-BFGS)
        # converges cleanly with it -- "random slopes where they converge".
        for method in (("lbfgs",) if re_formula else ("lbfgs", "powell", "nm")):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                try:
                    model = smf.mixedlm(formula, d, groups=d["participant_id"], re_formula=re_formula)
                    res = model.fit(reml=True, method=[method], maxiter=5000)
                except Exception as exc:  # singular fits etc.
                    last_err = exc
                    continue
            if not res.converged:
                last_err = "did not converge (%s)" % method
                continue
            msgs = sorted({str(x.message) for x in w
                           if any(s in str(x.message).lower()
                                  for s in ("converge", "boundary", "singular"))})
            finite = bool(np.all(np.isfinite(res.bse_fe)) and np.all(res.bse_fe < 1e3 * (1 + np.abs(res.fe_params))))
            singular = any("singular" in m.lower() for m in msgs)
            fits.append((res, method, msgs, finite and not singular))
            if not msgs and finite:
                break           # a clean fit: no need to try further optimisers
        if re_formula:
            ok = [f for f in fits if not f[2] and f[3]]
        else:
            ok = [f for f in fits if f[3]] or [f for f in fits if np.all(np.isfinite(f[0].bse_fe))]
        if ok:
            res, method, msgs, _ = max(ok, key=lambda f: f[0].llf)
            return res, "%s [REML, %s]" % (label, method), msgs
    raise RuntimeError("mixed model failed: %s (%s)" % (formula, last_err))


def _coef(res, name, sd_total=None):
    b = float(res.fe_params[name])
    se = float(res.bse_fe[name])
    z = b / se
    p = float(2 * stats.norm.sf(abs(z)))
    lo, hi = b - 1.959964 * se, b + 1.959964 * se
    out = {"term": name, "estimate": b, "se": se, "ci_low": lo, "ci_high": hi,
           "z": z, "p": p}
    if sd_total:
        out.update({"d": b / sd_total, "d_ci_low": lo / sd_total, "d_ci_high": hi / sd_total})
    return out


def _sd_total(res):
    var_u = float(res.cov_re.iloc[0, 0])
    return float(np.sqrt(var_u + res.scale))


def _wald_joint(res, names):
    b = res.fe_params[names].values
    V = res.cov_params().loc[names, names].values
    chi2 = float(b @ np.linalg.solve(V, b))
    return chi2, len(names), float(stats.chi2.sf(chi2, len(names)))


def _model_meta(res, label, warns, d):
    return {"random_effects": label, "n_obs": int(res.nobs),
            "n_participants": int(d["participant_id"].nunique()),
            "converged": bool(res.converged), "warnings": warns,
            "var_participant": float(res.cov_re.iloc[0, 0]), "var_residual": float(res.scale)}


def fit_h1(t):
    d = _prep(t[t["woa"].notna()])
    res, label, warns = _fit_mixed("woa ~ rel_c + ai_correct + C(block)", d)
    c = _coef(res, "rel_c", _sd_total(res))
    return {"id": "H1", "outcome": "WoA (untransformed)", "model": "woa ~ rel_c + ai_correct + C(block) + (1|pid)",
            "test": "Reliability (80 % - 60 %)", "predicted_sign": +1, **c,
            "meta": _model_meta(res, label, warns, d),
            "other_terms": {k: _coef(res, k) for k in res.fe_params.index if k != "rel_c"}}


def fit_h1_truncated(t):
    d = _prep(t[t["woa_trunc"].notna()])
    res, label, warns = _fit_mixed("woa_trunc ~ rel_c + ai_correct + C(block)", d)
    return {"id": "H1-trunc", **_coef(res, "rel_c", _sd_total(res)),
            "meta": _model_meta(res, label, warns, d)}


def fit_h2(blocks):
    d = _prep(blocks[blocks["trust"].notna()])
    # three observations per person: a random slope on block is not identifiable
    res, label, warns = _fit_mixed("trust ~ rel_c * C(block)", d, try_slope=False)
    names = ["rel_c:C(block)[T.1]", "rel_c:C(block)[T.2]"]
    chi2, df, p = _wald_joint(res, names)
    c = _coef(res, "rel_c:C(block)[T.2]", _sd_total(res))
    c["p_contrast"] = c["p"]
    c["p"] = p
    return {"id": "H2", "outcome": "Trust (TiA, 1-7)", "model": "trust ~ rel_c * C(block) + (1|pid)",
            "test": "Reliability x Block (Wald chi2, %d df); estimate = divergence block 3 vs 1" % df,
            "predicted_sign": 0, "chi2": chi2, "df": df, **c,
            "meta": _model_meta(res, label, warns, d),
            "block1_rel_effect": _coef(res, "rel_c")}


def fit_h3(t):
    d = _prep(t[t["post_error"] & t["woa"].notna()])
    res, label, warns = _fit_mixed("woa ~ place_c * rel_c", d)
    c = _coef(res, "place_c", _sd_total(res))
    sens, *_ = _fit_mixed("woa ~ place_c * rel_c + ai_correct", d)
    trunc, *_ = _fit_mixed("woa_trunc ~ place_c * rel_c", d)
    confound = _current_error_share(d)
    return {"id": "H3", "outcome": "Post-error WoA", "model": "woa ~ place_c * rel_c + (1|pid), post-error trials",
            "test": "Placement (propagating - absorbed)", "predicted_sign": -1, **c,
            "meta": _model_meta(res, label, warns, d),
            "interaction": _coef(res, "place_c:rel_c"),
            "sensitivity_with_ai_correct": _coef(sens, "place_c"),
            "sensitivity_truncated_woa": _coef(trunc, "place_c"),
            "post_error_trials_that_are_errors": confound}


def _current_error_share(d):
    """Share of post-error trials that are themselves AI-error trials, by
    condition x placement. Propagating weeks cluster (7-8-9, 14-15-16), so a
    trial after a propagating error is far more often an error trial too --
    a design-induced confound for H3/H4b that the registered formula does
    not adjust for (see the `sensitivity_with_ai_correct` entries)."""
    g = d.groupby(["condition", "post_error_placement"])["ai_correct"]
    return {"%s/%s" % k: {"n": int(v.size), "share_ai_error": float(1 - v.mean())}
            for k, v in g}


def fit_h4(t):
    d = _prep(t)
    res, label, warns = _fit_mixed("decision_error ~ rel_c + C(block)", d)
    a = {"id": "H4a", "outcome": "Decision error |Q_final - Q*|",
         "model": "decision_error ~ rel_c + C(block) + (1|pid)",
         "test": "Reliability (80 % - 60 %)", "predicted_sign": -1,
         **_coef(res, "rel_c", _sd_total(res)), "meta": _model_meta(res, label, warns, d)}
    dp = _prep(t[t["post_error"]])
    res_b, label_b, warns_b = _fit_mixed("decision_error ~ rel_c * place_c", dp)
    b = {"id": "H4b", "outcome": "Post-error decision error",
         "model": "decision_error ~ rel_c * place_c + (1|pid), post-error trials",
         "test": "Reliability x Placement (gap widens after propagating)", "predicted_sign": -1,
         **_coef(res_b, "rel_c:place_c", _sd_total(res_b)),
         "meta": _model_meta(res_b, label_b, warns_b, dp)}
    sens, *_ = _fit_mixed("decision_error ~ rel_c * place_c + ai_correct", dp)
    b["sensitivity_with_ai_correct"] = _coef(sens, "rel_c:place_c")
    b["post_error_trials_that_are_errors"] = _current_error_share(dp)
    return a, b


def fit_cost_blocks(t):
    agg = (t.groupby(["participant_id", "condition", "block"], as_index=False)
             .agg(cost=("cost", "sum"), stockouts=("stockout", "sum")))
    d = _prep(agg)
    res, label, warns = _fit_mixed("cost ~ rel_c + C(block)", d, try_slope=False)
    return {"id": "H4-cost (secondary)", "outcome": "Block cost (sum of 10 weeks)",
            **_coef(res, "rel_c", _sd_total(res)), "meta": _model_meta(res, label, warns, d)}


# ------------------------------------------------------------ binomial GLMM
def glmm_logit_ghq(y, X, groups, n_nodes=30):
    """Random-intercept logistic regression by maximum likelihood.

    L_i = integral prod_j Bernoulli(y_ij | x_ij b + u) N(u; 0, s^2) du,
    approximated by Gauss-Hermite quadrature. Returns beta, SEs (inverse
    numerical Hessian of the log-likelihood), sigma and the log-likelihood.
    """
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    codes, uniq = pd.factorize(pd.Series(groups))
    order = np.argsort(codes, kind="stable")
    y, X, codes = y[order], X[order], codes[order]
    starts = np.r_[0, np.flatnonzero(np.diff(codes)) + 1]
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    logw = np.log(weights / np.sqrt(np.pi))

    def negll(theta):
        beta, log_s = theta[:-1], theta[-1]
        eta = X @ beta
        u = np.sqrt(2.0) * np.exp(log_s) * nodes          # (K,)
        lin = eta[:, None] + u[None, :]                    # (n, K)
        ll_obs = y[:, None] * lin - np.logaddexp(0, lin)   # Bernoulli log-lik
        ll_cluster = np.add.reduceat(ll_obs, starts, axis=0)   # (G, K)
        return -float(np.sum(np.logaddexp.reduce(ll_cluster + logw[None, :], axis=1)))

    start_beta = sm.Logit(y, X).fit(disp=0).params
    theta0 = np.r_[start_beta, np.log(0.5)]
    opt = optimize.minimize(negll, theta0, method="BFGS", options={"gtol": 1e-6, "maxiter": 2000})
    H = approx_hess(opt.x, negll)
    cov = np.linalg.inv(H)
    return {"beta": opt.x[:-1], "se": np.sqrt(np.diag(cov))[:-1], "sigma": float(np.exp(opt.x[-1])),
            "loglik": -float(opt.fun), "converged": bool(opt.success), "cov": cov[:-1, :-1],
            "n_groups": len(uniq)}


def fit_ar(t):
    d = _prep(t[t["AR"].notna()])
    X = pd.DataFrame({"Intercept": 1.0, "rel_c": d["rel_c"],
                      "block1": (d["block"] == 1).astype(float),
                      "block2": (d["block"] == 2).astype(float)})
    # block sum-coded for the reliability main effect: averaged over blocks
    X["block1"] -= 1 / 3
    X["block2"] -= 1 / 3
    X["rel_c:block1"] = X["rel_c"] * X["block1"]
    X["rel_c:block2"] = X["rel_c"] * X["block2"]
    names = list(X.columns)
    fit = glmm_logit_ghq(d["AR"].values, X.values, d["participant_id"].values)
    i = names.index("rel_c")
    b, se = float(fit["beta"][i]), float(fit["se"][i])
    lo, hi = b - 1.959964 * se, b + 1.959964 * se
    ib = [names.index("rel_c:block1"), names.index("rel_c:block2")]
    bb = fit["beta"][ib]
    Vb = fit["cov"][np.ix_(ib, ib)]
    chi2 = float(bb @ np.linalg.solve(Vb, bb))

    # robustness: GEE exchangeable, and the VB mixed GLM
    robust = {}
    # Composition check: AR pools CA trials (success = RAIR-type) and IA
    # trials (success = RSR-type). The 80 % condition has more CA trials by
    # design, so a reliability "effect" can arise from the case mix alone.
    Xc = X.copy()
    Xc["case_IA_c"] = d["case_IA"].astype(float).values - 0.5
    fc = glmm_logit_ghq(d["AR"].values, Xc.values, d["participant_id"].values)
    bc, sec = float(fc["beta"][i]), float(fc["se"][i])
    robust["sensitivity_with_case_type"] = {
        "estimate": bc, "se": sec, "ci_low": bc - 1.959964 * sec, "ci_high": bc + 1.959964 * sec,
        "p": float(2 * stats.norm.sf(abs(bc / sec))),
        "share_IA_by_condition": {c: float(v) for c, v in d.groupby("condition")["case_IA"].mean().items()}}
    try:
        gee = sm.GEE(d["AR"].values, X.values, groups=d["participant_id"].values,
                     family=sm.families.Binomial(), cov_struct=sm.cov_struct.Exchangeable()).fit()
        robust["gee_exchangeable"] = {"estimate": float(gee.params[i]), "se": float(gee.bse[i]),
                                      "p": float(gee.pvalues[i])}
    except Exception as exc:  # pragma: no cover
        robust["gee_exchangeable"] = {"error": str(exc)}
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        exog_vc = pd.get_dummies(d["participant_id"]).astype(float).values
        vb = BinomialBayesMixedGLM(d["AR"].values, X.values, exog_vc, np.zeros(exog_vc.shape[1], int)).fit_vb()
        robust["bayes_mixed_glm_vb"] = {"posterior_mean": float(vb.fe_mean[i]),
                                        "posterior_sd": float(vb.fe_sd[i])}
    except Exception as exc:  # pragma: no cover
        robust["bayes_mixed_glm_vb"] = {"error": str(exc)}

    return {"id": "AR", "outcome": "Appropriate reliance (binary, Schemmer CA/IA trials)",
            "model": "logit P(AR=1) ~ rel_c * C(block, sum) + (1|pid)  [ML, Gauss-Hermite 30]",
            "test": "Reliability (log-odds, 80 % - 60 %)", "predicted_sign": 0,
            "term": "rel_c", "estimate": b, "se": se, "ci_low": lo, "ci_high": hi,
            "z": b / se, "p": float(2 * stats.norm.sf(abs(b / se))),
            "odds_ratio": float(np.exp(b)), "or_ci_low": float(np.exp(lo)), "or_ci_high": float(np.exp(hi)),
            "interaction_wald_chi2": chi2, "interaction_p": float(stats.chi2.sf(chi2, 2)),
            "meta": {"method": "ML GLMM, non-adaptive Gauss-Hermite quadrature (30 nodes)",
                     "n_obs": int(len(d)), "n_participants": fit["n_groups"],
                     "sigma_participant": fit["sigma"], "converged": fit["converged"],
                     "terms": dict(zip(names, map(float, fit["beta"])))},
            "robustness": robust}


# ------------------------------------------------------------ Holm + table
def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - k) * p))
        adj[name] = running
    return adj


def run(measures: dict) -> dict:
    t, blocks = measures["trials"], measures["blocks"]
    h1 = fit_h1(t)
    h2 = fit_h2(blocks)
    h3 = fit_h3(t)
    h4a, h4b = fit_h4(t)
    ar = fit_ar(t)
    tests = {r["id"]: r for r in (h1, h2, h3, h4a, h4b, ar)}
    adj = holm({k: tests[k]["p"] for k in FAMILY})
    for k in FAMILY:
        r = tests[k]
        r["p_holm"] = adj[k]
        sign_ok = r["predicted_sign"] == 0 or np.sign(r["estimate"]) == r["predicted_sign"]
        r["supported"] = bool(adj[k] < ALPHA and sign_ok)
    return {"family": list(FAMILY), "alpha": ALPHA, "tests": tests,
            "secondary": {"H1_truncated_woa": fit_h1_truncated(t), "cost_block_level": fit_cost_blocks(t)}}


def _fmt_p(p):
    return "< .001" if p < 0.001 else ("%.3f" % p).lstrip("0")


def table2(results: dict, banner: str | None = None) -> str:
    rows = []
    for k in results["family"]:
        r = results["tests"][k]
        est = "%.3f [%.3f, %.3f]" % (r["estimate"], r["ci_low"], r["ci_high"])
        if k == "AR":
            est += "; OR %.2f [%.2f, %.2f]" % (r["odds_ratio"], r["or_ci_low"], r["or_ci_high"])
        elif "d" in r:
            est += "; d = %.2f" % r["d"]
        rows.append("| %s | %s | %s | %s | %s |" % (
            k, r["test"], est, _fmt_p(r["p_holm"]), "Yes" if r["supported"] else "No"))
    head = []
    if banner:
        head += ["> **%s**" % banner, ""]
    head += ["| Hypothesis | Test | Estimate [95 % CI] | p (Holm) | Supported |",
             "|---|---|---|---|---|"]
    family = ", ".join(results["family"])
    notes = ["", f"Holm correction across {family}. Supported = p(Holm) < .05 and estimate "
             "in the predicted direction (H2, AR non-directional). Estimates are model "
             "coefficients: reliability coded 80 % - 60 %, placement propagating - absorbed; "
             "d = b / sqrt(var_participant + var_residual)."]
    return "\n".join(head + rows + notes) + "\n"


def to_json(results: dict) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(results, indent=2, default=default)
