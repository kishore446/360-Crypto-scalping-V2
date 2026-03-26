"""Tests for per-pair config profiles (PR_02)."""

from config import PAIR_PROFILES, PAIR_TIER_MAP
from src.pair_manager import classify_pair_tier


class TestPairProfile:
    def test_major_tier_exists(self):
        assert "MAJOR" in PAIR_PROFILES
        assert PAIR_PROFILES["MAJOR"].tier == "MAJOR"

    def test_midcap_tier_exists(self):
        assert "MIDCAP" in PAIR_PROFILES
        assert PAIR_PROFILES["MIDCAP"].tier == "MIDCAP"

    def test_altcoin_tier_exists(self):
        assert "ALTCOIN" in PAIR_PROFILES
        assert PAIR_PROFILES["ALTCOIN"].tier == "ALTCOIN"

    def test_btcusdt_is_major(self):
        profile = classify_pair_tier("BTCUSDT")
        assert profile.tier == "MAJOR"
        assert profile.momentum_threshold_mult == 0.8
        assert profile.spread_max_mult == 0.5

    def test_ethusdt_is_major(self):
        profile = classify_pair_tier("ETHUSDT")
        assert profile.tier == "MAJOR"

    def test_dogeusdt_is_altcoin(self):
        profile = classify_pair_tier("DOGEUSDT")
        assert profile.tier == "ALTCOIN"
        assert profile.kill_zone_hard_gate is True
        assert profile.momentum_threshold_mult == 2.0

    def test_solusdt_is_midcap(self):
        profile = classify_pair_tier("SOLUSDT")
        assert profile.tier == "MIDCAP"

    def test_unknown_high_volume_is_major(self):
        profile = classify_pair_tier("XYZUSDT", volume_24h_usd=600_000_000)
        assert profile.tier == "MAJOR"

    def test_unknown_mid_volume_is_midcap(self):
        profile = classify_pair_tier("XYZUSDT", volume_24h_usd=100_000_000)
        assert profile.tier == "MIDCAP"

    def test_unknown_low_volume_is_altcoin(self):
        profile = classify_pair_tier("XYZUSDT", volume_24h_usd=1_000_000)
        assert profile.tier == "ALTCOIN"

    def test_case_insensitive_lookup(self):
        profile = classify_pair_tier("btcusdt")
        assert profile.tier == "MAJOR"

    def test_altcoin_rsi_levels(self):
        profile = classify_pair_tier("PEPEUSDT")
        assert profile.rsi_ob_level == 65.0
        assert profile.rsi_os_level == 35.0

    def test_major_rsi_levels(self):
        profile = classify_pair_tier("BTCUSDT")
        assert profile.rsi_ob_level == 75.0
        assert profile.rsi_os_level == 25.0

    def test_pair_tier_map_completeness(self):
        """All mapped pairs should resolve to valid profiles."""
        for sym, tier in PAIR_TIER_MAP.items():
            assert tier in PAIR_PROFILES, f"{sym} maps to unknown tier {tier}"
