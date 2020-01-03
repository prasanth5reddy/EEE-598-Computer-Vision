import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
import datetime

from models.lenet import LeNet
from utils.utils import *


def main():
    # Hyper parameters
    batch_size = 128
    learning_rate = 0.001
    momentum = 0.9
    epochs = 100
    gpu = True
    load_saved_model = False

    # print hyper parameters
    print(f'batch size : {batch_size}')
    print(f'learning rate : {learning_rate}')
    print(f'momentum : {momentum}')
    print(f'epochs: {epochs}')
    print(f'GPU: {gpu}')
    print(f'load saved model: {load_saved_model}')
    print()

    # set up GPU device
    device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

    # Loading CIFAR10
    # set dataset path here
    download_path = '/home/pbuddare/EEE_598/CNN/data'
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_loader, test_loader = prepare_cifar10_data(download_path, transform, transform, batch_size)
    classes = prepare_classes()
    print()

    # get some training data
    data_iter = iter(train_loader)
    images, labels = data_iter.next()

    # show images
    show_images(make_grid(images[:4]))
    # print labels
    print(' '.join(classes[labels[j]] for j in range(4)))

    # initialise network
    net = LeNet()
    net.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Train the network
    loss_list_train, loss_list_test = train_model(net, criterion, optimizer, train_loader, test_loader, device, epochs)

    # save the model
    PATH = './cifar10_lenet' + str(datetime.datetime.now().time()).replace(':', '').replace('.', '') + '.pth'
    torch.save(net.state_dict(), PATH)

    # if load saved model
    if load_saved_model:
        net = LeNet()
        net.to(device)
        net.load_state_dict(torch.load(PATH))

    # Test the network
    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    images = images.to(device)
    labels = labels.to(device)

    # show images
    show_images(make_grid(images[:4].cpu()), name='test')
    # print labels
    print('\nGround Truth:', ' '.join(classes[labels[j]] for j in range(4)))

    # predict few samples
    outputs = net(images)
    _, pred = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(classes[pred[j]] for j in range(4)))

    # get total accuracy
    test_accuracy = get_accuracy(net, test_loader, device)
    print(f'\nAccuracy on test data: {test_accuracy}%')

    # get class wise accuracy
    class_correct, class_total = get_class_accuracy(net, test_loader, device)
    print()
    for i in range(10):
        print(f'Accuracy of {classes[i]}: {class_correct[i] * 100 / class_total[i]}%')

    # Accuracy on test data: 64.77 %


if __name__ == '__main__':
    main()
