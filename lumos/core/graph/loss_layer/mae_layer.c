#include "mae_layer.h"

Layer *make_mae_layer(int group)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = MAE;
    l->group = group;

    l->initialize = init_mae_layer;
    l->forward = forward_mae_layer;
    l->backward = backward_mae_layer;

    l->initializegpu = init_mae_layer_gpu;
    l->forwardgpu = forward_mae_layer_gpu;
    l->backwardgpu = backward_mae_layer_gpu;

    l->weightinit = NULL;
    l->weightinitgpu = NULL;

    l->update = NULL;
    l->updategpu = NULL;

    l->saveweights = NULL;
    l->saveweightsgpu = NULL;

    l->freelayer = free_mae_layer;
    l->freelayergpu = free_mae_layer_gpu;

    fprintf(stderr, "Mae             Layer    :    [output=%4d]\n", 1);
    return l;
}

void init_mae_layer(Layer *l, int w, int h, int c, int subdivision)
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

    l->output = calloc(subdivision*l->outputs, sizeof(float));
    l->delta = calloc(subdivision*l->inputs, sizeof(float));

    fprintf(stderr, "Mae             Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_mae_layer(Layer l, int num)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_o = i*l.outputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *output = l.output+offset_o;
        float *truth = l.truth+offset_t;
        matrix_subtract_cpu(truth, input, l.inputs, l.workspace);
        absolute(l.workspace, l.inputs, 1);
        sum_cpu(l.workspace, l.inputs, output);
        multy_cpu(output, l.outputs, 1/(float)l.group, 1);
    }
    sum_cpu(l.output, l.outputs*num, l.loss);
    multy_cpu(l.loss, 1, (float)1/num, 1);
}

void backward_mae_layer(Layer l, float rate, int num, float *n_delta)
{
    for (int i = 0; i < num; ++i){
        int offset_i = i*l.inputs;
        int offset_t = i*l.group;
        float *input = l.input+offset_i;
        float *delta_l = l.delta+offset_i;
        float *truth = l.truth+offset_t;
        matrix_subtract_cpu(input, truth, l.inputs, delta_l);
        delta_absolute(delta_l, l.inputs, 1);
        multy_cpu(delta_l, l.inputs, (float)1/l.group, 1);
    }
}

void free_mae_layer(Layer l)
{
    free(l.output);
    free(l.delta);
}

void delta_absolute(float *data, int len, int offset)
{
    for (int i = 0; i < len; i+=offset){
        if (data[i] >= 0) data[i] = 1;
        else data[i] = -1;
    }
}

void absolute(float *data, int len, int offset)
{
    for (int i = 0; i < len; i+=offset){
        data[i] = fabs(data[i]);
    }
}