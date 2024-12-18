#include "avgpool_layer_gpu.h"

void init_avgpool_layer_gpu(Layer *l, int w, int h, int c, int subdivision)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = (h + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_w = (w + 2 * l->pad - l->ksize) / l->stride + 1;
    l->output_c = l->input_c;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->workspace_size = 0;

    cudaMalloc((void**)&l->output, subdivision*l->outputs*sizeof(float));
    cudaMalloc((void**)&l->delta, subdivision*l->inputs*sizeof(float));

    fprintf(stderr, "Avg Pooling     Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_avgpool_layer_gpu(Layer l, int num)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        avgpool_gpu(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, output);
    }
}

void backward_avgpool_layer_gpu(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        avgpool_gradient_gpu(delta_l, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, delta_n);
    }
}

void free_avgpool_layer_gpu(Layer l)
{
    cudaFree(l.output);
    cudaFree(l.delta);
}
