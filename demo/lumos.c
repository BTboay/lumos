#include "lenet5_cifar10.h"
#include "lenet5_mnist.h"

int main()
{
    lenet5_cifar10("cpu", NULL);
    lenet5_cifar10_detect("cpu", "./LuWeights");
    // lenet5_mnist("cpu", NULL);
    // lenet5_mnist_detect("cpu", "./LuWeights");
    return 0;
}