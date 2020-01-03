import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time


# function to show image
def show_images(img, name='train'):
    img = img / 2 + 0.5
    img_np = img.numpy()
    # plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.imsave('./results/' + name + '.png', np.transpose(img_np, (1, 2, 0)))
    plt.show()


# prepare dataset
def prepare_cifar10_data(path, transform_train, transform_test, batch_size):
    train_set = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# prepare classes
def prepare_classes():
    return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Train the network
def train_model(net, criterion, optimizer, train_loader, test_loader, device, epochs):
    print(f'Running on {device}')
    loss_list_train, loss_list_test = [], []
    for epoch in range(1, epochs + 1):
        start = time.time()
        running_loss_train, running_loss_test = 0, 0
        num_of_samples_train, num_of_samples_test = 0, 0
        for i, data in enumerate(train_loader, 0):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.cpu().item()
            num_of_samples_train += y.cpu().shape[0]

        with torch.no_grad():
            for data in test_loader:
                X, y = data
                X = X.to(device)
                y = y.to(device)
                loss = criterion(net(X), y)
                running_loss_test += loss.cpu().item()
                num_of_samples_test += y.cpu().shape[0]

        loss_epoch_train = running_loss_train / num_of_samples_train
        loss_epoch_test = running_loss_test / num_of_samples_test
        print(f'Epoch: [{epoch}/{epochs}], Run Time: {round(time.time() - start, 4)} Sec')
        print(f'Train loss: {round(loss_epoch_train, 4)}, Test loss: {round(loss_epoch_test, 4)}')
        loss_list_train.append(loss_epoch_train)
        loss_list_test.append(loss_epoch_test)
    return loss_list_train, loss_list_test


# Get accuracy
def get_accuracy(net, data_loader, device):
    # get total accuracy
    correct, total = 0, 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, pred = torch.max(outputs.data, 1)
            total += labels.cpu().size(0)
            correct += (pred == labels).sum().cpu().item()
    return correct * 100 / total


# Get class wise accuracy
def get_class_accuracy(net, data_loader, device):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            labels = labels.cpu()
            c = c.cpu()
            for i in range(labels.shape[0]):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    return class_correct, class_total


def train_model_alt(net, criterion, optimizer, train_loader, test_loader, device, epochs):
    print(f'Running on {device}')
    loss_list_train, acc_list_train, acc_list_test = [], [], []
    for epoch in range(1, epochs + 1):
        start = time.time()
        running_loss_train, num_of_samples_train = 0, 0
        for i, data in enumerate(train_loader, 0):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            running_loss_train += loss.cpu().item()
            num_of_samples_train += y.cpu().shape[0]

        loss_train = running_loss_train / num_of_samples_train
        train_acc, test_acc = get_accuracy(net, train_loader, device), get_accuracy(net, test_loader, device)
        print(f'Epoch: [{epoch}/{epochs}], Run Time: {round(time.time() - start, 4)} Sec')
        print(f'Train loss: {round(loss_train, 4)},', end=' ')
        print(f'Train accuracy: {round(train_acc, 4)}, Test accuracy: {round(test_acc, 4)}')
        acc_list_train.append(train_acc)
        acc_list_test.append(test_acc)

    return acc_list_train, acc_list_test
