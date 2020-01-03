import torch
from torchvision import transforms, datasets, utils as vutils
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import time


def prepare_celeba_data(path, batch_size, image_size, workers):
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    data_set = datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=workers)


def prepare_square_data(path, batch_size, image_size, workers):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    data_set = datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=workers)


def prepare_dolphin_data(path, transform, batch_size, workers):
    data_set = datasets.ImageFolder(root=path, transform=transform)
    return DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=workers)


# function to show images
def show_images(imgs, fig_size, num_of_images, title, name):
    # plt.figure(figsize=fig_size)
    # plt.axis("off")
    # plt.title(title)
    # plt.imshow(np.transpose(vutils.make_grid(imgs[:num_of_images], padding=2, normalize=True), (1, 2, 0)))
    plt.imsave('./results/' + name + '.png',
               np.transpose(vutils.make_grid(imgs[:num_of_images], padding=2, normalize=True), (1, 2, 0)))
    plt.show()


# From the DCGAN Paper, these weight initializations are given.
def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1, 0.02)
        nn.init.constant_(m.bias.data, 0)


# define noise for generator input
def noise(batch_size, nz, gpu):
    noise = Variable(torch.randn(batch_size, nz, 1, 1))
    return noise.cuda() if gpu else noise


# generate real labels
def real_label(batch_size, gpu):
    data = Variable(torch.ones(batch_size, 1, 1, 1))
    return data.cuda() if gpu else data


# generate fake labels
def fake_label(batch_size, gpu):
    data = Variable(torch.zeros(batch_size, 1, 1, 1))
    return data.cuda() if gpu else data


# Train discriminator
def train_discriminator(X, X_dash, net_D, loss, optim_D, gpu):
    optim_D.zero_grad()

    real_Y = net_D(X)
    fake_Y = net_D(X_dash)

    loss_D = loss(real_Y, real_label(X.size(0), gpu)) + loss(fake_Y, fake_label(X.size(0), gpu))
    loss_D.backward()

    optim_D.step()
    return torch.sum(loss_D).item()


# Train generator
def train_generator(X_dash, net_D, loss, optim_G, gpu):
    optim_G.zero_grad()

    fake_Y = net_D(X_dash)

    loss_G = loss(fake_Y, real_label(X_dash.size(0), gpu))
    loss_G.backward()

    optim_G.step()
    return torch.sum(loss_G).item()


# Train GAN
def train_gan(data_loader, net_G, net_D, loss, optim_D, optim_G, batch_size, nz, epochs, gpu):
    list_loss_G, list_loss_D = [], []
    for epoch in range(1, epochs + 1):
        start = time.time()
        sum_loss_G, sum_loss_D, num_of_samples = 0, 0, 0
        for i, (X, _) in enumerate(data_loader):
            # Train Discriminator
            X = X.cuda() if gpu else X
            Z = noise(X.size(0), nz, gpu)
            X_dash = net_G(Z)
            # Don't want to update weights of generator while training discriminator. Hence, detach()
            loss_D = train_discriminator(X, X_dash.detach(), net_D, loss, optim_D, gpu)
            sum_loss_D += loss_D

            # Train Generator
            loss_G = train_generator(X_dash, net_D, loss, optim_G, gpu)
            sum_loss_G += loss_G

            num_of_samples += batch_size

            # print(f'Epoch: [{epoch}/{num_epochs}], Batch Number: [{i + 1}/{len(dataloader)}]')
            # print(f'Discriminator Loss: {loss_D}, Generator Loss: {loss_G}')

        loss_dis, loss_gen = sum_loss_D / num_of_samples, sum_loss_G / num_of_samples
        print(f'Epoch: [{epoch}/{epochs}], Run Time: {round(time.time() - start, 4)} Sec')
        print(f'Discriminator Loss: {round(loss_dis, 4)}, Generator Loss: {round(loss_gen, 4)}')
        list_loss_D.append(loss_dis)
        list_loss_G.append(loss_gen)

    return list_loss_G, list_loss_D
