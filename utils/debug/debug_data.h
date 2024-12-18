#ifndef DEBUG_DATA_H
#define DEBUG_DATA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "text_f.h"

#ifdef __cplusplus
extern "C" {
#endif

void debug_cu_data_f(float *data, int h, int w, int c, char *path);

#ifdef __cplusplus
}
#endif
#endif
