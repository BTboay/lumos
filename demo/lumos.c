#include "lenet5_cifar10.h"
#include "lenet5_mnist.h"
#include "dvc.h"
#include "image.h"
#include "lenet5_dvc.h"
#include "alexnet.h"

int main()
{
    lenet5_mnist("gpu", NULL);
    lenet5_mnist_detect("gpu", "./LuWeights");
    return 0;
}