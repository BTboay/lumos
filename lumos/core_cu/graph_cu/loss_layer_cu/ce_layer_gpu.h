#ifndef CE_LAYER_GPU_H
#define CE_LAYER_GPU_H

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

void init_ce_layer_gpu(Layer *l, int w, int h, int c, int subdivision);
void forward_ce_layer_gpu(Layer l, int num);
void backward_ce_layer_gpu(Layer l, float rate, int num, float *n_delta);

void free_ce_layer_gpu(Layer l);

void cross_entropy_gpu(float *data_a, float *data_b, int len, float *space);
void delta_cross_entropy_gpu(float *data_a, float *data_b, int len, float *space);

#ifdef __cplusplus
}
#endif
#endif
