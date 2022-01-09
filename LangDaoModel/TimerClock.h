//
// Created by Administrator on 2021/4/13.
//

#ifndef CPP11TIMER_TIMERCLOCK_H
#define CPP11TIMER_TIMERCLOCK_H

#include <chrono>

class TimerClock {
public:
    TimerClock();
    virtual ~TimerClock() = default;
    void update();
    double getSecond();
    double getMilliSecond();
    double getMicroSecond();

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
};
//
// Created by Administrator on 2021/4/13.
//


using namespace std::chrono;

TimerClock::TimerClock() {
    update();
}

void TimerClock::update() {
    mStart = high_resolution_clock::now();
}

double TimerClock::getSecond() {
    return getMicroSecond() * 0.000001;
}

double TimerClock::getMilliSecond() {
    return getMicroSecond() * 0.001;
}

double TimerClock::getMicroSecond() {
    return duration_cast<microseconds>(high_resolution_clock::now() - mStart).count();
}



#endif //CPP11TIMER_TIMERCLOCK_H