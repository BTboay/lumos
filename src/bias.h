#ifndef BIAS_H
#define BIAS_H

#include <stdio.h>
#include <stdlib.h>

#include "tensor.h"
#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif

void add_bias(Tensor *ts, Array *bias, int n, int size);

#ifdef __cplusplus
}
#endif

#endif