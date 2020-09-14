#include "Thor.h"

#include <stdio.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

using std::pair;
using std::set;
using std::string;
using std::vector;

TEST(Optional, Works) {
    Optional<int> maybeInt;
    assert(!maybeInt.isPresent());
    Optional<long> maybeLong(100);
    assert(maybeLong == 100);
    maybeInt = Optional<int>(100);
    assert(maybeInt.isPresent());
    assert(!maybeInt.isEmpty());
    assert(maybeInt == maybeLong);

    maybeInt = 200;
    assert(maybeInt == 200);
    maybeInt.set(300);
    assert(maybeInt == Optional<int>(300));
    assert(maybeInt.get() == 300);

    maybeInt.clear();
    assert(!maybeInt.isPresent());
    assert(maybeInt.isEmpty());

    maybeInt = Optional<int>(50);
    assert(maybeInt == 50);

    maybeInt = Optional<int>(Optional<int>(1000));
    assert(maybeInt == 1000);

    Optional<string> maybeString("a string");
    assert(maybeString.get() == "a string");
    string otherString(maybeString);
    assert(otherString == "a string");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
