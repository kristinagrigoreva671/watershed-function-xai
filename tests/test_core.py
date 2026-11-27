import numpy as np
import pandas as pd
import pytest

from wfx.budyko import fit_omega, fu_evaporative_index
from wfx.features import FEATURES, TARGETS
from wfx.io import load_attributes
from wfx.signatures import baseflow_index, baseflow_lyne_hollick, slope_fdc, stream_elasticity


def _daily(values, start="2000-10-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def test_baseflow_bounded_by_streamflow():
    rng = np.random.default_rng(0)
    q = np.abs(rng.gamma(0.6, 2.0, 1500)) + 0.05
    bf = baseflow_lyne_hollick(q)
    assert bf.shape == q.shape
    assert np.all(bf >= 0) and np.all(bf <= q + 1e-12)


def test_constant_flow_is_all_baseflow():
    assert baseflow_index(_daily(np.full(800, 2.0))) == pytest.approx(1.0)


def test_flashier_series_has_lower_bfi():
    t = np.arange(1500)
    smooth = 1 + 0.5 * np.sin(2 * np.pi * t / 365)
    flashy = smooth + 8 * (t % 23 == 0)
    assert baseflow_index(_daily(flashy)) < baseflow_index(_daily(smooth))


def test_fdc_slope_zero_for_constant_flow():
    assert slope_fdc(_daily(np.full(400, 1.0))) == pytest.approx(0.0)


def test_elasticity_of_proportional_response_is_one():
    rng = np.random.default_rng(1)
    days = pd.date_range("1990-10-01", "2005-09-30", freq="D")
    wy = np.where(days.month >= 10, days.year + 1, days.year)
    annual_p = {y: rng.uniform(2, 4) for y in np.unique(wy)}
    p = pd.Series([annual_p[y] for y in wy], index=days)
    q = 0.4 * p
    assert stream_elasticity(q, p) == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("omega", [1.5, 2.6, 4.0])
@pytest.mark.parametrize("phi", [0.4, 1.0, 2.5])
def test_fu_omega_round_trip(phi, omega):
    e = float(fu_evaporative_index(phi, omega))
    assert fit_omega(phi, e) == pytest.approx(omega, rel=1e-6)


def test_fit_omega_outside_limits_is_nan():
    assert np.isnan(fit_omega(0.5, 0.8))   # E/P above the energy limit
    assert np.isnan(fit_omega(1.0, -0.1))  # Q > P


def test_attributes_load():
    df = load_attributes()
    assert len(df) == 671
    missing = [c for c in FEATURES + [t for t in TARGETS if t != "fu_omega"] if c not in df]
    assert not missing
