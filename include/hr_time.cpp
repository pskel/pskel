
void hrt_start(hr_timer_t *hr_timer)
{
#ifdef __WIN32
    QueryPerformanceCounter(&hr_timer->start_tm) ;
#else
    clock_gettime(CLOCK_REALTIME, &hr_timer->start_tm);
#endif
}

void hrt_stop(hr_timer_t *hr_timer)
{
#ifdef __WIN32
    QueryPerformanceCounter(&hr_timer->end_tm) ;
#else
    clock_gettime(CLOCK_REALTIME, &hr_timer->end_tm);
#endif
}

#ifdef __WIN32
double LIToSecs(LARGE_INTEGER * L)
{
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency( &frequency );
    return ((double)L->QuadPart /(double)frequency.QuadPart);
}
#endif

double hrt_elapsed_time(hr_timer_t *hr_timer)
{
#ifdef __WIN32
    LARGE_INTEGER time;
    time.QuadPart = hr_timer->end_tm.QuadPart - hr_timer->start_tm.QuadPart;
    return LIToSecs( &time) ;
#else
    return  ((double)(hr_timer->end_tm.tv_sec - hr_timer->start_tm.tv_sec))+((double)(hr_timer->end_tm.tv_nsec - hr_timer->start_tm.tv_nsec)/1000000000L);
#endif
}
