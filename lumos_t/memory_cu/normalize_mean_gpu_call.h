#ifndef NORMALIZE_MEAN_GPU_CALL_H
#define NORMALIZE_MEAN_GPU_CALL_H

#include "session.h"
#include "manager.h"
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_normalize_mean_gpu(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif