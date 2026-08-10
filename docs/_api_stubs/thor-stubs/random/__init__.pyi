

class FullPeriodRandom:
    def __init__(self, period: int, synchronized: bool = False) -> None:
        """
        Random generator that produces each integer in ``[0, period)`` exactly once
        per cycle, in randomized order.

        After completing a full cycle, it reseeds itself and begins a new randomized cycle.

        Parameters
        ----------
        period : int
            Number of distinct values in each cycle. Must be nonzero.
        synchronized : bool, default False
            Whether access should be mutex-protected for multi-threaded use.
        """

    def get_random_number(self) -> int:
        """
        Return the next random number in the current full-period cycle.

        Returns
        -------
        int
            A value in ``[0, period)``.
        """

    def reseed(self, seed_value: int | None = None) -> None:
        """
        Reseed the generator and start a new randomized cycle.

        Parameters
        ----------
        seed_value : Optional[int], default None
            Optional explicit seed for the internal state. Otherwise the seed will be set to a random value (using entropy, time, etc).
        """

    def get_seed(self) -> int:
        """Return the seed that is currently in use for the period."""
