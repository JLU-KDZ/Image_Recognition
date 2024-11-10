import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
from ResNet import ResNet
import matplotlib.pyplot as plt

# 训练设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "train_out.txt"

def train():
    batch_size = 100
    # 训练集
    cifar_train = datasets.CIFAR10(
        root='cifar',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]))
    cifar_train = DataLoader(cifar_train,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

    # 测试集
    cifar_test = datasets.CIFAR10(
        root='cifar',
        train=False,
        transform=transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]))
    cifar_test = DataLoader(cifar_test,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    # 定义模型-ResNet
    model = ResNet()
    model = torch.load("net.pth")
    model.to(device)

    # 定义损失函数和优化方式
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.001)

    # 存储每个epoch的平均损失和准确率
    loss_sums = []
    accuracies = []

    # 训练网络
    for epoch in range(15):
        f = open(path, "a+")
        model.train()  # 训练模式
        loss_sum = 0
        for batchidx, (data, label) in enumerate(cifar_train):
            data = data.to(device)
            label = label.to(device)
            predict = model(data)
            loss = criterion(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            if (batchidx % 20 == 0):
                print("Epoch_num:", epoch,"  batch_num:",batchidx, '  item-loss:',loss.item())
                print("Epoch_num:", epoch,"  batch_num:",batchidx, '  item-loss:',loss.item(),file=f)

        average_loss = loss_sum / len(cifar_train)
        loss_sums.append(average_loss)
        print("Epoch_num:", epoch, '  training-mean-loss:', average_loss)
        print("Epoch_num:", epoch, '  training-mean-loss:', average_loss, file=f)

        model.eval()  # 测试模式
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for data, label in cifar_test:
                data = data.to(device)
                label = label.to(device)
                predict = model(data)

                pred = predict.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += data.size(0)
            acc = total_correct / total_num
            accuracies.append(acc)
            print("Epoch_num:", epoch, '  test_acc:', acc)
            print("Epoch_num:", epoch, '  test_acc:', acc, file=f)
            torch.save(model, "net.pth")
            f.close()

    return loss_sums, accuracies

if __name__ == '__main__':
    loss_sums, accuracies = train()
    epochs = range(1, 16)  # 假设有5个epoch

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_sums, 'r-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, 'b-', label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()