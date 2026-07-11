from __future__ import annotations

from speculative.controller.lut import LUT2D


def test_lookup_basic(simple_config):
    lut = LUT2D(simple_config)
    assert lut.lookup(bs=30, ctx=100) == 3
    assert lut.lookup(bs=30, ctx=2000) == 3
    assert lut.lookup(bs=128, ctx=100) == 0
    assert lut.lookup(bs=128, ctx=2000) == 3


def test_bs_clamps_high(simple_config):
    lut = LUT2D(simple_config)
    assert lut.lookup(bs=1024, ctx=100) == 0
    assert lut.lookup(bs=1024, ctx=2000) == 3


def test_bs_clamps_low(simple_config):
    lut = LUT2D(simple_config)
    assert lut.lookup(bs=0, ctx=100) == 3


def test_tier_of_boundaries(simple_config):
    lut = LUT2D(simple_config)
    assert lut.tier_of(30, 512) == lut.tier_of(30, 0)
    assert lut.tier_of(30, 513) == lut.tier_of(30, 2000)
    assert lut.tier_of(60, 100) == lut.tier_of(30, 100)
    assert lut.tier_of(61, 100) == lut.tier_of(200, 100)


def test_k_at_tier(simple_config):
    lut = LUT2D(simple_config)
    tier = lut.tier_of(30, 100)
    assert lut.k_at(tier) == 3


def test_gemma_dose_response_row(gemma_h100_config):
    lut = LUT2D(gemma_h100_config)
    assert lut.lookup(bs=200, ctx=150) == 0
    assert lut.lookup(bs=200, ctx=460) == 3
    assert lut.lookup(bs=200, ctx=970) == 3
    assert lut.lookup(bs=200, ctx=1990) == 3
