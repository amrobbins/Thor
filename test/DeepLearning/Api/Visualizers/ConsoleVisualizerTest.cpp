#include "Thor.h"

#include "gtest/gtest.h"

#include <bits/stdc++.h>

using namespace std;
using namespace Thor;

// TEST(ConsoleVisualizer, ManualTest) {

int main() {
    ConsoleVisualizer consoleVisualizer;

    consoleVisualizer.startUI();

    mutex mtx;
    mtx.lock();
    mtx.lock();
}
