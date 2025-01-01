#ifndef LENET5_H
#define LENET5_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void lenet5(char *type, char *path);
void lenet5_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
