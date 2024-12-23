#include "lenet5_cifar10.h"
#include "lenet5_mnist.h"
#include "dvc.h"

int main()
{
    lenet5_cifar10("gpu", NULL);
    lenet5_cifar10_detect("gpu", "./LuWeights");
    // lenet5_mnist("gpu", NULL);
    // lenet5_mnist_detect("gpu", "./LuWeights");
    // dvc("gpu", NULL);
    // dvc_detect("gpu", "./LuWeights");
    return 0;
}