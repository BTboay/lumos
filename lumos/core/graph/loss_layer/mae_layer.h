#ifndef MAE_LAYER_H
#define MAE_LAYER_H

#include "layer.h"
#include "cpu.h"
#include "gemm.h"

#include "mae_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_mae_layer(int group);

void init_mae_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_mae_layer(Layer l, int num);
void backward_mae_layer(Layer l, float rate, int num, float *n_delta);

void free_mae_layer(Layer l);

void absolute(float *data, int len, int offset);
void delta_absolute(float *data, int len, int offset);

#ifdef __cplusplus
}
#endif

#endif