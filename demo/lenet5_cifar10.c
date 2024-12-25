#include "lenet5_cifar10.h"

void lenet5_cifar10(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(128, 3, 1, 1, 1, 1, "leaky");
    Layer *l2 = make_convolutional_layer(128, 3, 1, 1, 1, 1, "leaky");
    Layer *l3 = make_convolutional_layer(128, 3, 1, 1, 1, 1, "leaky");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_dropout_layer(0.5);
    Layer *l6 = make_convolutional_layer(256, 3, 1, 1, 1, 1, "leaky");
    Layer *l7 = make_convolutional_layer(256, 3, 1, 1, 1, 1, "leaky");
    Layer *l8 = make_convolutional_layer(256, 3, 1, 1, 1, 1, "leaky");
    Layer *l9 = make_maxpool_layer(2, 2, 0);
    Layer *l10 = make_dropout_layer(0.5);
    Layer *l11 = make_convolutional_layer(512, 3, 1, 1, 1, 1, "leaky");
    Layer *l12 = make_convolutional_layer(512, 3, 1, 1, 1, 1, "leaky");
    Layer *l13 = make_dropout_layer(0.5);
    Layer *l14 = make_convolutional_layer(10, 1, 1, 1, 1, 1, "leaky");
    Layer *l15 = make_im2col_layer();
    Layer *l16 = make_connect_layer(100, 1, "leaky");
    Layer *l17 = make_connect_layer(10, 1, "leaky");
    Layer *l18 = make_softmax_layer(10);
    Layer *l19 = make_mse_layer(10);
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
    append_layer2grpah(g, l12);
    append_layer2grpah(g, l13);
    append_layer2grpah(g, l14);
    append_layer2grpah(g, l15);
    append_layer2grpah(g, l16);
    append_layer2grpah(g, l17);
    append_layer2grpah(g, l18);
    append_layer2grpah(g, l19);
    Session *sess = create_session(g, 32, 32, 3, 10, type, path);
    set_train_params(sess, 5000, 128, 128, 0.4);
    init_session(sess, "./data/cifar10/train.txt", "./data/cifar10/train_label.txt");
    train(sess);
}

