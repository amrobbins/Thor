#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "Utilities/Random/FullPeriodRandom.h"

namespace nb = nanobind;
using namespace nb::literals;

void bind_full_period_random(nb::module_ &m) {
    auto full_period_random = nb::class_<FullPeriodRandom>(m, "FullPeriodRandom");
    full_period_random.attr("__module__") = "thor.random";

    full_period_random.def(
        "__init__",
        [](FullPeriodRandom *self, uint64_t period, bool synchronized) { new (self) FullPeriodRandom(period, synchronized); },
        "period"_a,
        "synchronized"_a = false,
        R"nbdoc(
            Random generator that produces each integer in ``[0, period)`` exactly once
            per cycle, in randomized order.

            After completing a full cycle, it reseeds itself and begins a new randomized cycle.

            Parameters
            ----------
            period : int
                Number of distinct values in each cycle. Must be nonzero.
            synchronized : bool, default False
                Whether access should be mutex-protected for multi-threaded use.
        )nbdoc");

    full_period_random.def("get_random_number",
                           &FullPeriodRandom::getRandomNumber,
                           R"nbdoc(
            Return the next random number in the current full-period cycle.

            Returns
            -------
            int
                A value in ``[0, period)``.
        )nbdoc");

    full_period_random.def(
        "reseed",
        [](FullPeriodRandom &self, std::optional<uint64_t> seed_value) {
            if (seed_value.has_value())
                self.reseed(Optional<uint64_t>(seed_value.value()));
            else
                self.reseed(Optional<uint64_t>::empty());
        },
        "seed_value"_a.none() = nb::none(),
        R"nbdoc(
            Reseed the generator and start a new randomized cycle.

            Parameters
            ----------
            seed_value : Optional[int], default None
                Optional explicit seed for the internal state. Otherwise the seed will be set to a random value (using entropy, time, etc).
        )nbdoc");

    full_period_random.def("get_seed",
                           &FullPeriodRandom::getSeed,
                           R"nbdoc(
            Return the seed that is currently in use for the period.
        )nbdoc");
}
