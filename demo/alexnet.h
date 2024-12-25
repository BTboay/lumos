#ifndef ALEXNET_H
#define ALEXNET_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void alexnet(char *type, char *path);
void alexnet_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
