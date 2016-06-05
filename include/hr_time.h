/*
  * time/time.h
  *
  *  Global definitions for some useful functions related to time.
  *
  * version:   1.0.0
  * date:      15 September 2011
  * author:    (rcor) Rodrigo Caetano de Oliveira Rocha
  *
*/
#ifndef __HR_TIME
#define __HR_TIME

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __WIN32
#include <windows.h>
#define __HR_TIMER_BASE_TYPE    LARGE_INTEGER
#else
#include <time.h>
#include <sys/time.h>
#define __HR_TIMER_BASE_TYPE    struct timespec
#endif

typedef struct __hr_timer{
   __HR_TIMER_BASE_TYPE start_tm;
   __HR_TIMER_BASE_TYPE end_tm;
} hr_timer_t;

void hrt_start(hr_timer_t *hr_timer);
void hrt_stop(hr_timer_t *hr_timer);
double hrt_elapsed_time(hr_timer_t *hr_timer);


#ifdef __cplusplus
}
#endif

#endif

#include "hr_time.cpp"
