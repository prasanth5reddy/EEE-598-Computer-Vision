import torch
import random
from torch import nn, optim
import datetime

from models.dcgan import Discriminator, Generator
from utils.utils import *


def main():
    # Set random seed for reproducibility
    manual_seed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manual_seed)
    print()
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    # Hyper parameters
    workers = 2
    batch_size = 128
    image_size = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    learning_rate = 0.0002
    beta1 = 0.5
    epochs = 250
    gpu = True
    load_saved_model = False

    # print hyper parameters
    print(f'number of workers : {workers}')
    print(f'batch size : {batch_size}')
    print(f'image size : {image_size}')
    print(f'number of channels : {nc}')
    print(f'latent dimension size : {nz}')
    print(f'generator feature map size : {ngf}')
    print(f'discriminator feature map size : {ndf}')
    print(f'learning rate : {learning_rate}')
    print(f'beta1 : {beta1}')
    print(f'epochs: {epochs}')
    print(f'GPU: {gpu}')
    print(f'load saved model: {load_saved_model}')
    print()

    # set up GPU device
    cuda = True if gpu and torch.cuda.is_available() else False

    # load dolphin dataset
    # dataset is downloaded from flickr using Image Downloader chrome extension.
    # Approximately 2200 images are downloaded
    download_path = '/home/pbuddare/EEE_598/GAN/data/Dolphin1'
    transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ])
    transform_aug = transforms.Compose([transforms.Resize((image_size, image_size)),
                                        # transforms.CenterCrop(image_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ])
    # data_loader_normal = prepare_dolphin_data(download_path, transform, batch_size, workers)
    data_loader_aug = prepare_dolphin_data(download_path, transform_aug, batch_size, workers)
    data_loader = data_loader_aug
    # show sample images
    show_images(next(iter(data_loader))[0], (8, 8), 40, 'Training images (Dolphin)', 'dolphin_train')

    # create generator and discriminator networks
    generator = Generator(nc, nz, ngf)
    discriminator = Discriminator(nc, ndf)
    if cuda:
        generator.cuda()
        discriminator.cuda()
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # loss function and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    # Train GAN
    loss_G, loss_D = train_gan(data_loader, generator, discriminator, criterion,
                               optimizer_d, optimizer_g, batch_size, nz, epochs, cuda)

    # save parameters
    current_time = str(datetime.datetime.now().time()).replace(':', '').replace('.', '') + '.pth'
    g_path = './Dolphin_G_' + current_time
    d_path = './Dolphin_D_' + current_time
    torch.save(generator.state_dict(), g_path)
    torch.save(discriminator.state_dict(), d_path)

    # generate and display fake images
    fake_imgs = generator(noise(batch_size, nz, cuda)).detach()
    show_images(fake_imgs.cpu(), (8, 8), 40, 'Fake images (Dolphin)', 'dolphin_fake')


if __name__ == '__main__':
    main()
