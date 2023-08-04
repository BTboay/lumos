#ifndef ROLL_MEAN_GPU_CALL_H
#define ROLL_MEAN_GPU_CALL_H

#include "session.h"
#include "manager.h"
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_roll_mean_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif
