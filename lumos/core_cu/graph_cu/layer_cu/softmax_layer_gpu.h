#ifndef SOFTMAX_LAYER_GPU_H
#define SOFTMAX_LAYER_GPU_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "cublas_v2.h"

#include <stdio.h>
#include <stdlib.h>

#include "gpu.h"
#include "layer.h"
#include "softmax_gpu.h"
#include "cpu_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

void forward_softmax_layer_gpu(Layer l, int num);
void backward_softmax_layer_gpu(Layer l, float rate, int num, float *n_delta);

#ifdef __cplusplus
}
#endif
#endif