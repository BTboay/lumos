#include "lenet5_dvc.h"

void lenet5_dvc(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, 0, "leaky");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, 0, "leaky");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, 0, "leaky");
    Layer *l6 = make_im2col_layer();
    Layer *l7 = make_connect_layer(84, 1, "leaky");
    Layer *l8 = make_connect_layer(2, 1, "leaky");
    Layer *l9 = make_softmax_layer(2);
    Layer *l10 = make_mse_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    append_layer2grpah(g, l9);
    append_layer2grpah(g, l10);
    Session *sess = create_session(g, 32, 32, 3, 2, type, path);
    set_train_params(sess, 50, 1, 1, 0.01);
    init_session(sess, "./build/path.txt", "./data/dogvscat/train_label.txt");
    train(sess);
}

void lenet5_dvc_detect(char*type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(6, 5, 1, 0, 1, 0, "leaky");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(16, 5, 1, 0, 1, 0, "leaky");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(120, 5, 1, 0, 1, 0, "leaky");
    Layer *l6 = make_im2col_layer();
    Layer *l7 = make_connect_layer(84, 1, "leaky");
    Layer *l8 = make_connect_layer(2, 1, "leaky");
    Layer *l9 = make_softmax_layer(2);
    Layer *l10 = make_mse_layer(2);
    append_layer2grpah(g, l1);
    append_layer2grpah(g, l2);
    append_layer2grpah(g, l3);
    append_layer2grpah(g, l4);
    append_layer2grpah(g, l5);
    append_layer2grpah(g, l6);
    append_layer2grpah(g, l7);
    append_layer2grpah(g, l8);
    append_layer2grpah(g, l9);
    append_layer2grpah(g, l10);
    Session *sess = create_session(g, 32, 32, 3, 2, type, path);
    set_detect_params(sess);
    init_session(sess, "./build/path.txt", "./data/dogvscat/train_label.txt");
    detect_classification(sess);
}
