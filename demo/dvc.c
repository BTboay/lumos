#include "dvc.h"

void dvc(char *type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(32, 3, 1, 1, 1, 0, "relu");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(32, 3, 1, 1, 1, 0, "relu");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(64, 3, 1, 1, 1, 0, "relu");
    Layer *l6 = make_maxpool_layer(2, 2, 0);
    Layer *l7 = make_dropout_layer(0.25);
    Layer *l8 = make_im2col_layer();
    Layer *l9 = make_connect_layer(64, 1, "relu");
    Layer *l10 = make_dropout_layer(0.5);
    Layer *l11 = make_connect_layer(2, 1, "relu");
    Layer *l12 = make_softmax_layer(2);
    Layer *l13 = make_mse_layer(2);
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
    Session *sess = create_session(g, 150, 150, 3, 2, type, path);
    set_train_params(sess, 50, 32, 32, 0.001);
    init_session(sess, "./build/path.txt", "./data/dogvscat/train_label.txt");
    train(sess);
}

void dvc_detect(char*type, char *path)
{
    Graph *g = create_graph();
    Layer *l1 = make_convolutional_layer(32, 3, 1, 0, 1, 0, "relu");
    Layer *l2 = make_maxpool_layer(2, 2, 0);
    Layer *l3 = make_convolutional_layer(32, 3, 1, 0, 1, 0, "relu");
    Layer *l4 = make_maxpool_layer(2, 2, 0);
    Layer *l5 = make_convolutional_layer(64, 3, 1, 0, 1, 0, "relu");
    Layer *l6 = make_maxpool_layer(2, 2, 0);
    Layer *l7 = make_dropout_layer(0.25);
    Layer *l8 = make_im2col_layer();
    Layer *l9 = make_connect_layer(64, 1, "relu");
    Layer *l10 = make_dropout_layer(0.5);
    Layer *l11 = make_connect_layer(2, 1, "relu");
    Layer *l12 = make_softmax_layer(2);
    Layer *l13 = make_mse_layer(2);
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
    Session *sess = create_session(g, 150, 150, 3, 2, type, path);
    set_detect_params(sess);
    init_session(sess, "./data/dogvscat/train.txt", "./data/dogvscat/train_label.txt");
    detect_classification(sess);
}
