import numpy as np
import cv2
import argparse

from utilities import save_samples
from gan import GAN

def build_gan(hyperparams):
    gan = GAN([128],[128],100,hyperparams)
    gan.train()

    return gan

def build_gan_celeba(hyperparams, path):
    gan = GAN([128], [128], 100, hyperparams, 'celeba_bw')
    gan.train(path)

    return gan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic MNIST Gan')
    parser.add_argument('--digit', type=int, default=2)
    parser.add_argument('--N_samples', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='mnist')
    args = parser.parse_args()

    hyperparams = {'lr': args.lr,
                   'epochs': args.epochs,
                   'batchSize': args.batchSize,
                   'number': args.digit}

    if args.dataset == 'mnist':
        trained_gan = build_gan(hyperparams)
    else:
        path_to_celeba = 'FILL THIS IN'
        trained_gan = build_gan_celeba(hyperparams, path_to_celeba)

    save_samples(trained_gan, args.N_samples)
