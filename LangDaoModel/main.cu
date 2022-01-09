#include"MVTfit.h"
#include "TimerClock.h"

int main() {
	TimerClock TC;
	TC.update();
	MVTLMfit LMfit;
	LMfit.start(TC);
	return 0;
}