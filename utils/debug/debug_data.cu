#include "debug_data.h"

void debug_cu_data_f(float *data, int h, int w, int c, char *path)
{
    FILE *buffer = fopen(path, "w");
    float *data_h = (float*)malloc(h*w*c*sizeof(float));
    cudaMemcpy(data_h, data, h*w*c*sizeof(float), cudaMemcpyDeviceToHost);
    for (int k = 0; k < c; ++k){
        for (int i = 0; i < h; ++i){
            for (int j = 0; j < w; ++j){
                fprintf(buffer, "%.3f ", data_h[k*h*w+i*w+j]);
            }
            fprintf(buffer, "\n");
        }
        fprintf(buffer, "\n");
    }
    free(data_h);
}
