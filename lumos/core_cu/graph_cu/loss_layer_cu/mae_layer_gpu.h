#ifndef MAE_LAYER_GPU_H
#define MAE_LAYER_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "cpu.h"
#include "gpu.h"
#include "layer.h"
#include "cpu_gpu.h"
#include "gemm_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void init_mae_layer_gpu(Layer *l, int w, int h, int c, int subdivision);
void forward_mae_layer_gpu(Layer l, int num);
void backward_mae_layer_gpu(Layer l, float rate, int num, float *n_delta);

void free_mae_layer_gpu(Layer l);

#ifdef __cplusplus
}
#endif
#endif
