#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PI 3.1415926

typedef float (*lrschedul) (float, int, int*, int, float, float, float, int, int);
typedef lrschedul LrSchedul;

typedef enum {
    SLR, MLR, ELR, CALR
} LrSchedulerType;

typedef struct lrscheduler{
    LrSchedulerType type;
    int num;
    int step_size;
    int *milestones;
    int T_max;
    float lr_min;
    float gamma;
    LrSchedul lrschedul;
} LrScheduler;

LrScheduler *make_lrscheduler(LrSchedulerType type, int num, int step_size, int *milestones, int T_max, float lr_min, float gamma);
float run_lrscheduler(LrScheduler *lrscheduler, float rate, float lr_max, int T_curr);

float step_scheduler(float rate, int step_size, int *milestones, int num, float lr_min, float lr_max, float gamma, int T_curr, int T_max);
float multistep_scheduler(float rate, int step_size, int *milestones, int num, float lr_min, float lr_max, float gamma, int T_curr, int T_max);
float exponential_scheduler(float rate, int step_size, int *milestones, int num, float lr_min, float lr_max, float gamma, int T_curr, int T_max);
float cosineannealing_scheduler(float rate, int step_size, int *milestones, int num, float lr_min, float lr_max, float gamma, int T_curr, int T_max);

#ifdef __cplusplus
}
#endif

#endif
