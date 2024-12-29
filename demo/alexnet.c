#include "alexnet.h"

void alexnet(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(16, 5, 1, 0, 1, 0, "leaky");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(32, 5, 1, 0, 1, 0, "leaky");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(64, 5, 1, 0, 1, 0, "leaky");
    Layer *l6 = make_maxpool_layer(2, 2, 0);
    Layer *l7 = make_convolutional_layer(128, 5, 1, 0, 1, 0, "leaky");
    Layer *l8 = make_maxpool_layer(2, 2, 0);
    Layer *l9 = make_im2col_layer();
    Layer *l10 = make_dropout_layer(0.2);
    Layer *l11 = make_connect_layer(240, 1, "leaky");
    Layer *l12 = make_connect_layer(84, 1, "leaky");
    Layer *l13 = make_dropout_layer(0.5);
    Layer *l14 = make_connect_layer(2, 1, "leaky");
    Layer *l15 = make_softmax_layer(2);
    Layer *l16 = make_mae_layer(2);
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
    Session *sess = create_session(g, 150, 150, 3, 2, type, path);
    set_train_params(sess, 200, 8, 8, 0.01);
    init_session(sess, "./build/path.txt", "./data/dogvscat/train_label.txt");
    train(sess);
}

