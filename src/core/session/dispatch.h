#ifndef DISPATCH_H
#define DISPATCH_H

#include "session.h"
#include "manager.h"
#include "graph.h"
#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

void session_run(Session *sess);

void forward_session(Session *sess);
void backward_session(Session *sess);
void update_session(Session *sess);

void create_run_scene(Session *sess, int h, int w, int c, char *dataset_list_file);
void init_run_scene(Session *sess, int epoch, int batch, int subdivision, char *weights_file);

#ifdef __cplusplus
}
#endif

#endif