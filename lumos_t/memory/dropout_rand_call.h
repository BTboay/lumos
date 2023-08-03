#ifndef DROPOUT_RAND_CALL_H
#define DROPOUT_RAND_CALL_H

#include "session.h"
#include "manager.h"
#include "dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

void call_dropout_rand(void **params, void **ret);

#ifdef __cplusplus
}
#endif

#endif