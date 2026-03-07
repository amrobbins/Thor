import thor
import pytest


def test_full_period_random_period_1_always_returns_0():
    rng = thor.random.FullPeriodRandom(1)

    for _ in range(20):
        assert rng.get_random_number() == 0


def test_full_period_random_period_10_returns_each_value_once_per_cycle():
    rng = thor.random.FullPeriodRandom(10)

    values = [rng.get_random_number() for _ in range(10)]

    assert len(values) == 10
    assert set(values) == set(range(10))


def test_full_period_random_two_cycles_still_cover_full_period():
    rng = thor.random.FullPeriodRandom(10)

    cycle1 = [rng.get_random_number() for _ in range(10)]
    cycle2 = [rng.get_random_number() for _ in range(10)]

    assert set(cycle1) == set(range(10))
    assert set(cycle2) == set(range(10))


def test_full_period_random_values_always_within_bounds():
    period = 37
    rng = thor.random.FullPeriodRandom(period)

    values = [rng.get_random_number() for _ in range(period * 3)]

    assert all(0 <= v < period for v in values)


def test_full_period_random_get_seed_returns_int():
    rng = thor.random.FullPeriodRandom(10)

    seed = rng.get_seed()

    assert isinstance(seed, int)
    assert seed >= 0


def test_full_period_random_reseed_with_explicit_seed_changes_seed():
    rng = thor.random.FullPeriodRandom(10)

    original_seed = rng.get_seed()
    rng.reseed(12345)
    new_seed = rng.get_seed()

    assert new_seed == 12345
    assert new_seed != original_seed


def test_full_period_random_reseed_without_seed_keeps_working():
    rng = thor.random.FullPeriodRandom(10)

    before = [rng.get_random_number() for _ in range(3)]
    rng.reseed()
    after = [rng.get_random_number() for _ in range(10)]

    assert len(before) == 3
    assert set(after) == set(range(10))


def test_full_period_random_reseed_explicit_seed_is_repeatable():
    period = 10
    rng1 = thor.random.FullPeriodRandom(period)
    rng2 = thor.random.FullPeriodRandom(period)

    rng1.reseed(999)
    rng2.reseed(999)

    seq1 = [rng1.get_random_number() for _ in range(3 * period)]
    seq2 = [rng2.get_random_number() for _ in range(3 * period)]

    assert seq1 == seq2
