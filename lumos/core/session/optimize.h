#ifndef OPTIMIZE_H
#define OPTIMIZE_H

#include <stdio.h>
#include <stdlib.h>

#include "session.h"

#ifdef __cplusplus
extern "C" {
#endif

void normalize_zscore(float *data, int num);

#ifdef __cplusplus
}
#endif

#endif
