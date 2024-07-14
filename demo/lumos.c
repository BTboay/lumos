#include "lenet5_mnist.h"

int main()
{
    lenet5_mnist("cpu", NULL);
    lenet5_mnist_detect("cpu", "./LuWeights");
    return 0;
}