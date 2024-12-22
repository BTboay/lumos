#ifndef DVC_H
#define DVC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void dvc(char *type, char *path);
void dvc_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif