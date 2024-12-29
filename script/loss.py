import matplotlib.pyplot  as plt

def GetLoss(path):
    loss = []
    with open(path, "r") as fp:
        for line in fp.readlines():
            if "[====================]" in line:
                loss.append(float(line.split(' ')[-1]))
    return loss

def DrawLoss(loss):
    x = [i for i in range(len(loss))]
    y = loss
    plt.figure(1)
    plt.plot(x, y, label="Training Loss")
    plt.title('LeNet5-MNIST loss')
    plt.legend()
    plt.savefig("./log/loss.png", dpi=600)

if __name__ == "__main__":
    loss = GetLoss("./log/lumos_t.log")
    DrawLoss(loss)