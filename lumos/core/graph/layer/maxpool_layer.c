#include "maxpool_layer.h"

Layer *make_maxpool_layer(int ksize, int stride, int pad)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = MAXPOOL;
    l->pad = pad;

    l->ksize = ksize;
    l->stride = stride;
    l->update = NULL;

    l->initialize = init_maxpool_layer;
    l->forward = forward_maxpool_layer;
    l->backward = backward_maxpool_layer;
    l->initialize_gpu = init_maxpool_layer_gpu;
    l->forward_gpu = forward_maxpool_layer_gpu;
    l->backward_gpu = backward_maxpool_layer_gpu;

    fprintf(stderr, "Max Pooling     Layer    :    [ksize=%2d]\n", l->ksize);
    return l;
}

void init_maxpool_layer(Layer *l, int w, int h, int c)
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

    l->output = calloc(l->outputs*l->subdivision, sizeof(float));
    l->delta = calloc(l->inputs*l->subdivision, sizeof(float));
    l->maxpool_index = calloc(l->outputs, sizeof(float));

    fprintf(stderr, "Max Pooling     Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_maxpool_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        int *index = l.maxpool_index + offset_o;
        maxpool(input, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, output, index);
    }
}

void backward_maxpool_layer(Layer l, float rate, int num)
{
    for (int i = 0; i < num; ++i)
    {
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        float *delta_l = l.delta + offset_i;
        float *delta_n = l.n_delta + offset_o;
        int *index = l.maxpool_index + offset_o;
        maxpool_gradient(delta_l, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, delta_n, index);
    }
}
