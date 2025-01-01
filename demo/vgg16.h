#ifndef VGG16_H
#define VGG16_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#ifdef __cplusplus
extern "C" {
#endif

void vgg16(char *type, char *path);
void vgg16_detect(char*type, char *path);

#ifdef __cplusplus
}
#endif
#endif
