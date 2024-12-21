#include "lenet5_cifar10.h"

void lenet5_cifar10(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(32, 5, 1, 2, 1, 0, "leaky");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(32, 5, 1, 2, 1, 0, "leaky");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(64, 5, 1, 2, 1, 0, "leaky");
    Layer *l6 = make_maxpool_layer(2, 2, 0);
    Layer *l7 = make_im2col_layer();
    Layer *l8 = make_connect_layer(64, 1, "leaky");
    Layer *l9 = make_connect_layer(10, 1, "leaky");
    Layer *l10 = make_softmax_layer(10);
    Layer *l11 = make_mse_layer(10);
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
    append_layer2grpah(g, l11);
    Session *sess = create_session(g, 32, 32, 3, 10, type, path);
    set_train_params(sess, 300, 16, 16, 0.001);
    init_session(sess, "./data/cifar10/train.txt", "./data/cifar10/train_label.txt");
    dynamic_learning_rate(sess, 100, 1);
    train(sess);
}

void lenet5_cifar10_detect(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(32, 5, 1, 2, 1, 0, "leaky");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(32, 5, 1, 2, 1, 0, "leaky");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(64, 5, 1, 2, 1, 0, "leaky");
    Layer *l6 = make_maxpool_layer(2, 2, 0);
    Layer *l7 = make_im2col_layer();
    Layer *l8 = make_connect_layer(64, 1, "leaky");
    Layer *l9 = make_connect_layer(10, 1, "leaky");
    Layer *l10 = make_softmax_layer(10);
    Layer *l11 = make_mse_layer(10);
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
    append_layer2grpah(g, l11);
    Session *sess = create_session(g, 32, 32, 3, 10, type, path);
    set_detect_params(sess);
    init_session(sess, "./data/cifar10/test.txt", "./data/cifar10/test_label.txt");
    detect_classification(sess);
}
