#include "session.h"

Session *create_session(Graph *graph, int h, int w, int c, int truth_num, char *type)
{
    Session *sess = malloc(sizeof(Session));
    sess->graph = graph;
    if (0 == strcmp(type, "gpu")){
        sess->coretype = GPU;
    } else {
        sess->coretype = CPU;
    }
    sess->height = h;
    sess->width = w;
    sess->channel = c;
    sess->truth_num = truth_num;
    return sess;
}

void init_session(Session *sess, char *data_path, char *label_path)
{
    bind_train_data(sess, data_path);
    bind_train_label(sess, label_path);
    init_graph(sess->graph, sess->width, sess->height, sess->channel, sess->coretype, sess->subdivision);
    create_workspace(sess);
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->input, sess->subdivision*sess->width*sess->height*sess->channel*sizeof(float));
        cudaMalloc((void**)&sess->truth, sess->subdivision*sess->truth_num*sizeof(float));
        cudaMalloc((void**)&sess->loss, sizeof(float));
    } else {
        sess->input = calloc(sess->subdivision*sess->width*sess->height*sess->channel, sizeof(float));
        sess->truth = calloc(sess->subdivision*sess->truth_num, sizeof(float));
        sess->loss = calloc(1, sizeof(float));
    }
    Graph *g = sess->graph;
    Node *layer = g->head;
    Layer *l;
    for (;;){
        if (layer){
            l = layer->l;
            l->truth = sess->truth;
            l->loss = sess->loss;
            l->workspace = sess->workspace;
        } else {
            break;
        }
        layer = layer->next;
    }
}

void bind_train_data(Session *sess, char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, '\n');
    int lines = index[0];
    char *data_path = NULL;
    sess->train_data_num = lines;
    sess->train_data_paths = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        data_path = tmp+index[i+1];
        strip(data_path, '\n');
        sess->train_data_paths[i] = data_path;
    }
    free(index);
    fprintf(stderr, "\nGet Train Data List From %s\n", path);
}

void bind_test_data(Session *sess, char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, '\n');
    int lines = index[0];
    char *data_path = NULL;
    sess->test_data_num = lines;
    sess->test_data_paths = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        data_path = tmp+index[i+1];
        strip(data_path, '\n');
        sess->test_data_paths[i] = data_path;
    }
    free(index);
    fprintf(stderr, "\nGet Test Data List From %s\n", path);
}

void bind_train_label(Session *sess, char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, '\n');
    int lines = index[0];
    char *label_path = NULL;
    sess->train_label_paths = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        label_path = tmp+index[i+1];
        strip(label_path, '\n');
        sess->train_label_paths[i] = label_path;
    }
    free(index);
    fprintf(stderr, "\nGet Label List From %s\n", path);
}

void bind_test_label(Session *sess, char *path)
{
    char *tmp = fget(path);
    int *index = split(tmp, '\n');
    int lines = index[0];
    char *label_path = NULL;
    sess->test_label_paths = malloc(lines*sizeof(char*));
    for (int i = 0; i < lines; ++i){
        label_path = tmp+index[i+1];
        strip(label_path, '\n');
        sess->test_label_paths[i] = label_path;
    }
    free(index);
    fprintf(stderr, "\nGet Label List From %s\n", path);
}

void set_train_params(Session *sess, int epoch, int batch, int subdivision, float learning_rate)
{
    sess->epoch = epoch;
    sess->batch = batch;
    sess->subdivision = subdivision;
    sess->learning_rate = learning_rate;
}

void create_workspace(Session *sess)
{
    Graph *graph = sess->graph;
    Node *layer = graph->head;
    Layer *l;
    int max = -1;
    for (;;){
        if (layer){
            l = layer->l;
            if (l->workspace_size > max) max = l->workspace_size;
        } else {
            break;
        }
        layer = layer->next;
    }
    if (max <= 0) return;
    if (sess->coretype == GPU){
        cudaMalloc((void**)&sess->workspace, max*sizeof(float));
    } else {
        sess->workspace = calloc(max, sizeof(float));
    }
}

void load_train_data(Session *sess, int index)
{
    int h[1], w[1], c[1];
    float *im;
    int offset_i = 0;
    float *input = (float*)calloc(sess->subdivision*sess->width*sess->height*sess->channel, sizeof(float));
    for (int i = index; i < index + sess->subdivision; ++i){
        char *data_path = sess->train_data_paths[i];
        im = load_image_data(data_path, w, h, c);
        resize_im(im, h[0], w[0], c[0], sess->height, sess->width, input + offset_i);
        offset_i += sess->height * sess->width * sess->channel;
        free(im);
    }
    if (sess->coretype == GPU){
        cudaMemcpy(sess->input, input, sess->subdivision*sess->width*sess->height*sess->channel*sizeof(float), cudaMemcpyHostToDevice);
    } else {
        memcpy(sess->input, input, sess->subdivision*sess->width*sess->height*sess->channel*sizeof(float));
    }
    free(input);
}

void load_train_label(Session *sess, int index)
{
    float *truth = calloc(sess->subdivision*sess->truth_num, sizeof(float));
    for (int i = index; i < index + sess->subdivision; ++i){
        float *truth_i = truth + (i - index) * sess->truth_num;
        char *label_path = sess->train_label_paths[i];
        strip(label_path, ' ');
        void **labels = load_label_txt(label_path);
        int *lindex = (int*)labels[0];
        char *tmp = (char*)labels[1];
        for (int j = 0; j < sess->truth_num; ++j){
            truth_i[j] = (float)atoi(tmp+lindex[j+1]);
        }
        free(lindex);
        free(tmp);
        free(labels);
    }
    if (sess->coretype == GPU){
        cudaMemcpy(sess->truth, truth, sess->truth_num*sess->subdivision*sizeof(float), cudaMemcpyHostToDevice);
    } else {
        memcpy(sess->truth, truth, sess->truth_num*sess->subdivision*sizeof(float));
    }
    free(truth);
}

void train(Session *sess)
{
    fprintf(stderr, "\nSession Start To Running\n");
    float rate = -sess->learning_rate / (float)sess->batch;
    float *loss = calloc(1, sizeof(float));
    clock_t start, final;
    double run_time = 0;
    for (int i = 0; i < sess->epoch; ++i){
        fprintf(stderr, "\n\nEpoch %d/%d\n", i + 1, sess->epoch);
        start = clock();
        int sub_epochs = (int)(sess->train_data_num / sess->batch);
        int sub_batchs = (int)(sess->batch / sess->subdivision);
        for (int j = 0; j < sub_epochs; ++j){
            for (int k = 0; k < sub_batchs; ++k){
                if (j * sess->batch + k * sess->subdivision + sess->subdivision > sess->train_data_num) break;
                load_train_data(sess, j * sess->batch + k * sess->subdivision);
                load_train_label(sess, j * sess->batch + k * sess->subdivision);
                forward_graph(sess->graph, sess->input, sess->coretype, sess->subdivision);
                backward_graph(sess->graph, rate, sess->coretype, sess->subdivision);
                final = clock();
                run_time = (double)(final - start) / CLOCKS_PER_SEC;
                if (sess->coretype == CPU) {
                    run_time /= 10;
                    loss[0] = sess->loss[0];
                } else{
                    cudaMemcpy(loss, sess->loss, sizeof(float), cudaMemcpyDeviceToHost);
                }
                progress_bar(j * sub_batchs + k + 1, sub_epochs * sub_batchs, run_time, loss[0]);
            }
            update_graph(sess->graph, sess->coretype);
        }
    }
    fprintf(stderr, "\n\nSession Training Finished\n");
}

void detect(Session *sess)
{

}
