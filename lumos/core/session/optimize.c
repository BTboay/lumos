#include "optimize.h"

LrScheduler *make_lrscheduler(LrSchedulerType type, int num, int step_size, int *milestones, int T_max, float lr_min, float gamma)
{
    LrScheduler *lrscheduler = malloc(sizeof(LrScheduler));
    lrscheduler->num = num;
    lrscheduler->step_size = step_size;
    lrscheduler->milestones = milestones;
    lrscheduler->T_max = T_max;
    lrscheduler->lr_min = lr_min;
    lrscheduler->gamma = gamma;
    switch (type){
        case SLR:
            lrscheduler->lrschedul = step_scheduler;
            break;
        case MLR:
            lrscheduler->lrschedul = multistep_scheduler;
            break;
        case ELR:
            lrscheduler->lrschedul = exponential_scheduler;
            break;
        case CALR:
            lrscheduler->lrschedul = cosineannealing_scheduler;
            break;
        default:
            lrscheduler->lrschedul = NULL;
            break;
    }
    return lrscheduler;
}

float run_lrscheduler(LrScheduler *lrscheduler, float rate, float lr_max, int T_curr)
{
    if (lrscheduler == NULL) return rate;
    float new_lr = -1;
    new_lr = lrscheduler->lrschedul(rate, lrscheduler->step_size, lrscheduler->milestones, lrscheduler->num, lrscheduler->lr_min, lr_max, lrscheduler->gamma, T_curr, lrscheduler->T_max);
    return new_lr;
}

float step_scheduler(float rate, int step_size, int *milestones, int num, float lr_min, float lr_max, float gamma, int T_curr, int T_max)
{
    float new_lr = -1;
    if ((T_curr+1) == step_size){
        new_lr = rate * gamma;
        return new_lr;
    }
    return rate;
}

float multistep_scheduler(float rate, int step_size, int *milestones, int num, float lr_min, float lr_max, float gamma, int T_curr, int T_max)
{
    float new_lr = -1;
    for (int i = 0; i < num; ++i){
        if ((T_curr+1) == milestones[i]){
            new_lr = rate * gamma;
            return new_lr;
        }
    }
    return rate;
}

float exponential_scheduler(float rate, int step_size, int *milestones, int num, float lr_min, float lr_max, float gamma, int T_curr, int T_max)
{
    float new_lr = -1;
    new_lr = rate * powf(gamma, T_curr);
    return new_lr;
}

float cosineannealing_scheduler(float rate, int step_size, int *milestones, int num, float lr_min, float lr_max, float gamma, int T_curr, int T_max)
{
    float new_lr = -1;
    float x = (T_curr / T_max)*PI;
    new_lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(x));
    return new_lr;
}
