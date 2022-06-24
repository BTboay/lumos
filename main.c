#include <stdio.h>
#include <stdlib.h>

#include "session.h"
#include "manager.h"
#include "dispatch.h"
#include "graph.h"
#include "layer.h"
#include "convolutional_layer.h"
#include "connect_layer.h"
#include "im2col_layer.h"
#include "maxpool_layer.h"
#include "avgpool_layer.h"

int main(int argc, char **argv)
{
    Graph *graph = create_graph("Lumos", 9);
    Layer *l1 = make_convolutional_layer(16, 3, 1, 1, 1, 1, "relu");
    Layer *l2 = make_avgpool_layer(2);
    Layer *l3 = make_convolutional_layer(16, 3, 1, 1, 1, 1, "relu");
    Layer *l4 = make_maxpool_layer(2);
    Layer *l5 = make_im2col_layer(1);
    Layer *l6 = make_connect_layer(128, 1, "relu");
    Layer *l7 = make_connect_layer(64, 1, "relu");
    Layer *l8 = make_connect_layer(32, 1, "relu");
    Layer *l9 = make_connect_layer(1, 1, "relu");
    Layer *l10 = make_im2col_layer(1);
    append_layer2grpah(graph, l1);
    append_layer2grpah(graph, l2);
    append_layer2grpah(graph, l3);
    append_layer2grpah(graph, l4);
    append_layer2grpah(graph, l5);
    append_layer2grpah(graph, l6);
    append_layer2grpah(graph, l7);
    append_layer2grpah(graph, l8);
    append_layer2grpah(graph, l9);
    append_layer2grpah(graph, l10);

    Session *sess = create_session();
    bind_graph(sess, graph);
    create_run_scene(sess, 28, 28, 1, 1, "./demo/xor/data.txt", "./demo/xor/label.txt");
    init_run_scene(sess, 20, 25, 5, NULL);
    session_run(sess);
}