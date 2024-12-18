#include "run_test.h"

int main(int argc, char **argv)
{
    run_by_benchmark_file("./lumos_t/benchmark/core/ops/im2col/im2col.json", argv);
    return 0;
}