#include "lenet5_cifar10.h"
#include "lenet5_mnist.h"
#include "dvc.h"
#include "image.h"
#include "lenet5_dvc.h"
#include "alexnet.h"
#include "xor.h"

int main()
{
    lenet5_cifar10("gpu", NULL);
    // lenet5_cifar10_detect("gpu", "./build/LW_f");
    return 0;
}