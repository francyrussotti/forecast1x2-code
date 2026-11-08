import numpy as np
import pandas as pd
from scipy.optimize import check_grad
from scipy.stats import poisson

from forecast1x2.model import N_GLOBAL, fit, neg_loglik, outcome_probs, predict_match, score_matrix, tau


def test_tau_hand_values():
    lam, mu, rho = 1.5, 1.1, -0.1
    assert np.isclose(tau(0, 0, lam, mu, rho), 1 - lam * mu * rho)
    assert np.isclose(tau(0, 1, lam, mu, rho), 1 + lam * rho)
    assert np.isclose(tau(1, 0, lam, mu, rho), 1 + mu * rho)
    assert np.isclose(tau(1, 1, lam, mu, rho), 1 - rho)
    assert tau(2, 1, lam, mu, rho) == 1 and tau(0, 3, lam, mu, rho) == 1


def test_tau_preserves_total_probability():
    # the DC correction redistributes mass among the four low scores: untruncated sum stays 1
    lam, mu, rho = 1.7, 0.9, -0.12
    g = np.arange(40)
    P = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu)) * tau(g[:, None], g[None, :], lam, mu, rho)
    assert np.isclose(P.sum(), 1.0, atol=1e-12)


def test_score_matrix_and_outcomes_sum_to_one():
    for lam, mu, rho in [(1.4, 1.1, -0.08), (3.0, 0.4, 0.05), (0.6, 0.6, -0.2)]:
        M = score_matrix(lam, mu, rho)
        assert M.shape == (11, 11) and (M >= 0).all()
        assert np.isclose(M.sum(), 1.0)
        assert np.isclose(sum(outcome_probs(M)), 1.0)


def test_rho_zero_is_independent_poisson():
    M = score_matrix(1.3, 0.8, 0.0, max_goals=30)
    assert np.isclose(M[0, 0], poisson.pmf(0, 1.3) * poisson.pmf(0, 0.8))


def _simulate(n_teams=12, rounds=12, seed=1):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    att = rng.normal(0, 0.3, n_teams); att -= att.mean()
    dfn = rng.normal(0, 0.25, n_teams); dfn -= dfn.mean()
    mu0, home, rho = 0.1, 0.25, -0.1
    rows, day = [], pd.Timestamp("2020-01-01")
    for r in range(rounds):
        for h in range(n_teams):
            for a in range(n_teams):
                if h == a:
                    continue
                lam, m = np.exp(mu0 + home + att[h] + dfn[a]), np.exp(mu0 + att[a] + dfn[h])
                M = score_matrix(lam, m, rho, max_goals=15)
                k = rng.choice(M.size, p=M.ravel())
                rows.append({"date": day + pd.Timedelta(days=r), "home": teams[h], "away": teams[a],
                             "hg": k // 16, "ag": k % 16})
    return pd.DataFrame(rows), teams, att, dfn, mu0, home, rho


def test_analytic_gradient():
    df, teams, *_ = _simulate(n_teams=6, rounds=2)
    idx = {t: i for i, t in enumerate(teams)}
    tier2 = np.arange(len(df)) % 3 == 0   # some matches flagged as second division, to check d/d delta
    args = (df.home.map(idx).to_numpy(), df.away.map(idx).to_numpy(), df.hg.to_numpy(), df.ag.to_numpy(),
            np.linspace(0.5, 1, len(df)), len(teams), 1.0, tier2)
    theta = np.random.default_rng(0).normal(0, 0.2, N_GLOBAL + 2 * (len(teams) - 1))
    theta[2] = -0.05
    err = check_grad(lambda t: neg_loglik(t, *args)[0], lambda t: neg_loglik(t, *args)[1], theta)
    assert err < 1e-5


def test_parameter_recovery_on_synthetic_data():
    df, teams, att, dfn, mu0, home, rho = _simulate()
    p = fit(df, df.date.max() + pd.Timedelta(days=1), xi=0.0, l2=0.0)
    order = [p.teams.index(t) for t in teams]
    assert np.abs(p.attack[order] - att).max() < 0.15
    assert np.abs(p.defence[order] - dfn).max() < 0.15
    assert abs(p.home - home) < 0.05
    assert abs(p.mu - mu0) < 0.05
    assert abs(p.rho - rho) < 0.06
    assert np.isclose(p.attack.sum(), 0) and np.isclose(p.defence.sum(), 0)


def test_second_division_offset_recovered():
    # two divisions with disjoint teams except two that played in both: delta must be recovered
    rng = np.random.default_rng(3)
    att = {f"A{i}": v for i, v in enumerate(rng.normal(0.2, 0.2, 8))} | {f"B{i}": v for i, v in enumerate(rng.normal(-0.2, 0.2, 8))}
    dfn = {t: -v / 2 for t, v in att.items()}
    mu0, home, delta = 0.1, 0.25, -0.15
    rows, day = [], pd.Timestamp("2020-01-01")
    for tier, pool in [(1, [f"A{i}" for i in range(8)] + ["B0"]), (2, [f"B{i}" for i in range(8)] + ["A7"])]:
        for r in range(8):
            for h in pool:
                for a in pool:
                    if h == a:
                        continue
                    off = delta if tier == 2 else 0.0
                    lam = np.exp(mu0 + off + home + att[h] + dfn[a])
                    m = np.exp(mu0 + off + att[a] + dfn[h])
                    rows.append({"date": day + pd.Timedelta(days=r), "tier": tier, "home": h, "away": a,
                                 "hg": rng.poisson(lam), "ag": rng.poisson(m)})
    df = pd.DataFrame(rows)
    p = fit(df, df.date.max() + pd.Timedelta(days=1), xi=0.0, l2=0.0)
    assert abs(p.delta - delta) < 0.07


def test_l2_shrinks_team_with_few_matches():
    df, *_ = _simulate()
    extra = pd.DataFrame([{"date": df.date.max(), "home": "New", "away": "T0", "hg": 5, "ag": 0}])
    df = pd.concat([df, extra], ignore_index=True)
    ref = df.date.max() + pd.Timedelta(days=1)
    free, shrunk = fit(df, ref, 0.0, l2=0.0), fit(df, ref, 0.0, l2=1.0)
    i = free.teams.index("New")
    assert abs(shrunk.attack[i]) < abs(free.attack[i])


def test_fit_rejects_future_matches():
    df, *_ = _simulate(n_teams=4, rounds=2)
    try:
        fit(df, df.date.min(), xi=0.002)
    except ValueError:
        return
    raise AssertionError("fit accepted matches after the reference date")


def test_predict_match_output():
    df, *_ = _simulate(n_teams=6, rounds=4)
    p = fit(df, df.date.max() + pd.Timedelta(days=1), xi=0.0)
    out = predict_match(p, "T0", "T1")
    assert np.isclose(out["p1"] + out["px"] + out["p2"], 1.0)
    assert out["xg_home"] > 0 and out["xg_away"] > 0
    assert out["top_score"].count("-") == 1


def test_gradient_where_tau_is_below_floor():
    # huge rates + rho > 0 + a 0-0: tau = 1 - lam*mu*rho < 0, the region that used to break the optimizer
    n = 3
    hi, ai = np.array([0, 1, 2, 0]), np.array([1, 2, 0, 2])
    x, y = np.array([0, 1, 0, 2]), np.array([0, 0, 1, 1])
    theta = np.zeros(N_GLOBAL + 2 * (n - 1))
    theta[0], theta[2] = 1.6, 0.15             # mu (lam ~ 5), rho
    theta[N_GLOBAL] = 0.4
    args = (hi, ai, x, y, np.ones(4), n)
    t00 = 1 - np.exp(1.6 + 0.4 - (-0.4)) * np.exp(1.6) * 0.15
    assert t00 < 0
    err = check_grad(lambda t: neg_loglik(t, *args)[0], lambda t: neg_loglik(t, *args)[1], theta)
    assert err < 1e-4 and np.isfinite(neg_loglik(theta, *args)[0])


def test_fit_converges_on_lopsided_data_with_tiny_penalty():
    rng = np.random.default_rng(7)
    rows, day = [], pd.Timestamp("2021-01-01")
    for r in range(6):
        for h, a, lh, la in [("Big", "Weak", 5.0, 0.2), ("Weak", "Big", 0.3, 4.0), ("Mid", "Weak", 2.0, 0.6),
                             ("Weak", "Mid", 0.8, 1.6), ("Big", "Mid", 2.5, 0.7), ("Mid", "Big", 0.9, 2.2)]:
            rows.append({"date": day + pd.Timedelta(days=7 * r), "home": h, "away": a,
                         "hg": rng.poisson(lh), "ag": rng.poisson(la)})
    rows.append({"date": day + pd.Timedelta(days=50), "home": "Big", "away": "Weak", "hg": 0, "ag": 0})
    df = pd.DataFrame(rows)
    p = fit(df, df.date.max() + pd.Timedelta(days=1), xi=0.006, l2=0.01)   # used to raise
    assert np.isfinite(p.attack).all() and -0.2 <= p.rho <= 0.2
