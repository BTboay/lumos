#include "lenet5.h"
#include "alexnet.h"
#include "xor.h"
#include "binary_f.h"

int main()
{
    lenet5("cpu", NULL);
    lenet5_detect("cpu", "./build/LW_f");
    return 0;
}