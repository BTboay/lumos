#include "mae_layer_gpu.h"

void init_mae_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = 1;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->workspace_size = l->inputs;

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));

    fprintf(stderr, "Mae             Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_mae_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *output = l.output+offset_o;
        float *truth = l.truth+offset_t;
        matrix_subtract_gpu(truth, input, l.inputs, l.workspace);
        absolute_gpu(l.workspace, l.inputs, 1);
        sum_gpu(l.workspace, l.inputs, output);
        multy_gpu(output, l.outputs, 1/(float)l.group, 1);
    }
    sum_gpu(l.output, l.outputs*num, l.loss);
    multy_gpu(l.loss, 1, (float)1/num, 1);
}

void backward_mae_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *delta_l = l.delta+offset_i;
        float *truth = l.truth+offset_t;
        matrix_subtract_gpu(input, truth, l.inputs, delta_l);
        delta_absolute_gpu(delta_l, l.inputs, 1);
        multy_gpu(delta_l, l.inputs, (float)1/l.group, 1);
    }
}

void free_mae_layer_gpu(Layer l)
{
    cudaFree(l.output);
    cudaFree(l.delta);
}

__global__ void absolute_kernel(float *data, int len, int offset)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x)*offset;
    if (index >= len) return;
    data[index] = fabs(data[index]);
}

__global__ void delta_absolute_kernel(float *data, int len, int offset)
{
    int index = (threadIdx.x + blockIdx.x * blockDim.x)*offset;
    if (index >= len) return;
    if (data[index] >= 0) data[index] = 1;
    else data[index] = -1;
}

void absolute_gpu(float *data, int len, int offset)
{
    absolute_kernel<<<(len+BLOCK-1)/BLOCK, BLOCK>>>(data, len, offset);
}

void delta_absolute_gpu(float *data, int len, int offset)
{
    delta_absolute_kernel<<<(len+BLOCK-1)/BLOCK, BLOCK>>>(data, len, offset);
}
