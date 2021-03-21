#include "Thor.h"

#include "gtest/gtest.h"

using namespace std;
using namespace Thor;

// TEST(ConsoleVisualizer, ManualTest) {

int main() {
    ConsoleVisualizer::instance().startUI();

    mutex mtx;
    mtx.lock();
    mtx.lock();
}
