#ifndef LENET5_DVC_H
#define LENET5_DVC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void lenet5_dvc(char *type, char *path);
void lenet5_dvc_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
