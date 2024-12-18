#include "run_test.h"
#include "im2col_call.h"

int main(int argc, char **argv)
{
    TestInterface FUNC = call_im2col;
    FILE *logfp = fopen("./log/logging", "w");
    run_by_benchmark_file("./lumos_t/benchmark/core/ops/im2col/im2col.json", FUNC, CPU, logfp);
    return 0;
}