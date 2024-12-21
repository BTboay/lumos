#include "optimize.h"

void normalize_zscore(float *data, int num)
{
    float mean = 0;
    float variance = 1;
    for (int i = 0; i < num; ++i){
        data[i] = (data[i] - mean) / variance;
    }
}
