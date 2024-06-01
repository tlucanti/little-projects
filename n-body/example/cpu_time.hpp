#ifndef _CPUTIME_HPP
#define _CPUTIME_HPP

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64

// MSVC defines this in winsock2.h!?
typedef struct timeval
{
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval *tp, struct timezone *tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970
    static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

    SYSTEMTIME system_time;
    FILETIME file_time;
    uint64_t time;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    time = ((uint64_t)file_time.dwLowDateTime);
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec = (long)((time - EPOCH) / 10000000L);
    tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
    return 0;
}
#else
#include <sys/time.h>
#endif

#include <sys/types.h>

// Return number of microseconds since 1.1.1970, in a 64 bit integer.

class CPUTime
{
private:
    double wctime;

    inline double readTime()
    {
        struct timeval tp;

        gettimeofday(&tp, nullptr);
        wctime = (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
        return wctime;
    }

public:
    CPUTime() : wctime(0.0) {}

    inline double start() { return readTime(); }
    inline double stop() { return readTime(); }
};

#endif
