#include "run_test.h"
#include "im2col_call.h"
#include "pooling_call.h"

int main(int argc, char **argv)
{
    TestInterface FUNC = call_maxpool;
    FILE *logfp = fopen("./log/logging", "w");
    run_by_benchmark_file("./lumos_t/benchmark/core/ops/pooling/maxpool.json", FUNC, CPU, logfp);
    return 0;
}