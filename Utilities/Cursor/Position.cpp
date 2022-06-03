#include "Position.h"

using namespace std;

void moveCursor(uint32_t row, uint32_t column) { printf("\033[%d;%dH", row, column); }