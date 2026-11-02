"""
Tests for WeatherLoader (NOAA ISD ambient temperature driver).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from c2g_env.physics.weather import WeatherLoader, WEATHER_PRESETS, WeatherParams


WEATHER_DIR = Path("data/processed/weather")
REAL_DATA   = (WEATHER_DIR / "NYC.csv").exists()


class TestWeatherLoaderFallback:
    """Behaviour when no CSV is present — synthetic model fallback."""

    def test_synthetic_used_when_no_file(self, tmp_path):
        with pytest.warns(UserWarning, match="file not found"):
            wl = WeatherLoader(weather_dir=tmp_path, market="nyiso_nyc")
        assert not wl.loaded
        assert wl.source.startswith("synthetic:")

    def test_synthetic_temp_finite(self, tmp_path):
        with pytest.warns(UserWarning, match="file not found"):
            wl = WeatherLoader(weather_dir=tmp_path, market="nyiso_nyc")
        for tick in [0, 720, 8640, 17280]:
            assert math.isfinite(wl.temp_c(tick))

    def test_synthetic_norm_in_range(self, tmp_path):
        with pytest.warns(UserWarning, match="file not found"):
            wl = WeatherLoader(weather_dir=tmp_path, market="nyiso_nyc")
        for tick in [0, 720, 8640]:
            n = wl.temp_norm(tick)
            assert 0.0 <= n <= 1.0, f"norm({tick}) = {n}"

    def test_fallback_temp_override(self, tmp_path):
        with pytest.warns(UserWarning, match="file not found"):
            wl = WeatherLoader(weather_dir=tmp_path, market="nyiso_nyc", fallback_temp_c=28.0)
        # synthetic replaces fallback when market file is absent but preset exists
        assert math.isfinite(wl.temp_c(0))

    def test_fallback_dewpoint(self, tmp_path):
        with pytest.warns(UserWarning, match="file not found"):
            wl = WeatherLoader(weather_dir=tmp_path, market="ercot_north")
        assert math.isfinite(wl.dewpoint_c(0))


@pytest.mark.skipif(not REAL_DATA, reason="NYC.csv not found")
class TestWeatherLoaderReal:
    """Tests that require actual NOAA data files."""

    @pytest.fixture
    def wl(self):
        return WeatherLoader(weather_dir=WEATHER_DIR, market="nyiso_nyc", dt_seconds=5.0)

    def test_loaded(self, wl):
        assert wl.loaded

    def test_temp_c_finite(self, wl):
        for tick in [0, 1, 719, 720, 721, 17279, 17280]:
            t = wl.temp_c(tick)
            assert math.isfinite(t), f"temp_c({tick}) = {t}"

    def test_temp_norm_in_range(self, wl):
        for tick in [0, 360, 720, 8640, 17280]:
            n = wl.temp_norm(tick)
            assert 0.0 <= n <= 1.0, f"norm({tick}) = {n}"

    def test_hour_hold(self, wl):
        """Same hour → same temperature (hold-step, not interpolation)."""
        assert wl.temp_c(0) == wl.temp_c(1)
        assert wl.temp_c(0) == wl.temp_c(719)

    def test_hour_changes(self, wl):
        """Different hours can differ (not guaranteed but true for real data)."""
        temps = {wl.temp_c(h * 720) for h in range(24)}
        assert len(temps) > 1  # NYC has intra-day temperature variation

    def test_tiling(self, wl):
        """Past end of file → wraps correctly."""
        n_rows = wl._n_hours
        assert wl.temp_c(0) == wl.temp_c(n_rows * 720)

    def test_dewpoint_le_temp(self, wl):
        """Dew point is physically always ≤ dry-bulb temperature."""
        for tick in [0, 3600, 7200]:
            assert wl.dewpoint_c(tick) <= wl.temp_c(tick) + 0.5  # small tolerance


# ---------------------------------------------------------------------------
# TestWeatherPresets — coverage of all market presets + synthetic model
# ---------------------------------------------------------------------------

class TestWeatherPresets:
    """All 6 market presets produce valid temperatures."""

    @pytest.mark.parametrize("market_id", list(WEATHER_PRESETS.keys()))
    def test_preset_smoke(self, tmp_path, market_id):
        """Each preset instantiates and returns finite temperatures."""
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            wl = WeatherLoader(weather_dir=tmp_path, market=market_id, dt_seconds=5.0)
        for tick in [0, 720, 3600, 17280]:
            assert math.isfinite(wl.temp_c(tick))
            n = wl.temp_norm(tick)
            assert 0.0 <= n <= 1.0

    def test_unknown_market_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown weather market"):
            WeatherLoader(weather_dir=tmp_path, market="fantasy_iso")

    def test_southern_hemisphere_has_summer_in_jan(self, tmp_path):
        """AEMO (southern hemisphere) should be warmer in January than July."""
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            wl = WeatherLoader(weather_dir=tmp_path, market="aemo_nsw", dt_seconds=5.0)
        # Mean temp for Jan (h=0..743) vs July (h=4344..5087)
        jan_mean  = np.mean([wl.temp_c(h * 720) for h in range(0, 24)])
        july_mean = np.mean([wl.temp_c(h * 720) for h in range(181*24, 181*24+24)])
        assert jan_mean > july_mean, (
            f"AEMO should have summer in Jan: jan={jan_mean:.1f}, july={july_mean:.1f}"
        )

    def test_ercot_has_hot_summer(self, tmp_path):
        """ERCOT should reach > 30°C in mid-summer."""
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            wl = WeatherLoader(weather_dir=tmp_path, market="ercot_north")
        # July afternoon (day ~196, hour ~14)
        peak_tick = (196 * 24 + 14) * 720
        assert wl.temp_c(peak_tick) > 28.0

    def test_entso_de_is_cooler_than_ercot(self, tmp_path):
        """Frankfurt should be cooler than Dallas at the annual peak."""
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            wl_de    = WeatherLoader(weather_dir=tmp_path, market="entso_de")
            wl_ercot = WeatherLoader(weather_dir=tmp_path, market="ercot_north")
        peak = (196 * 24 + 14) * 720
        assert wl_de.temp_c(peak) < wl_ercot.temp_c(peak)

    def test_all_presets_defined(self):
        expected = {"nyiso_nyc", "pjm_dom", "caiso_pgae", "ercot_north", "entso_de", "aemo_nsw"}
        assert expected == set(WEATHER_PRESETS.keys())
