#ifndef CE_LAYER_H
#define CE_LAYER_H

#include "layer.h"
#include "cpu.h"
#include "gemm.h"

#include "ce_layer_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif

Layer *make_ce_layer(int group);

void init_ce_layer(Layer *l, int w, int h, int c, int subdivision);
void forward_ce_layer(Layer l, int num);
void backward_ce_layer(Layer l, float rate, int num, float *n_delta);

void free_ce_layer(Layer l);

void cross_entropy(float *data_a, float *data_b, int len, float *space);
void delta_cross_entropy(float *data_a, float *data_b, int len, float *space);

#ifdef __cplusplus
}
#endif

#endif