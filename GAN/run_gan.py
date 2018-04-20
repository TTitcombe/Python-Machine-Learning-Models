import numpy as np
#import cv2
import argparse

from gan import GAN

def take_samples(model, N_samples, path='./generated_im'):
    anImg = gan.sample()
    for i in range(N_samples):
        path = path + str(i) + '.jpg'
        cv2.imwrite(path, anImg[i,:,:]*255)
        
def build_gan(hyperparams):
    gan = GAN(784, [128],[128],100,1, hyperparams)
    gan.train()
    
    return gan

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Basic MNIST Gan')
    parser.add_argument('--digit', type=int, default=2)
    parser.add_argument('--N_samples', type=str, default='1')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batchSize', type=int, default=64)
    args = parser.parse_args()

    hyperparams = {'lr': args.lr,
                   'epochs': args.epochs,
                   'batchSize': args.batchSize,
                   'number': args.digit}

    trained_gan = build_gan(hyperparams)

    take_samples(rained_gan, args.N_samples)
