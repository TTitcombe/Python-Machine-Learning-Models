# GAN 
### Requirements
* Python3 (should work for Python 2)
* Numpy 
* PIL (to save images)

gan.py provides a vanilla gan class - i.e. simple feedforward Descriminator and Generator set-up, adhering to the original GAN paper by Goodfellow et. al.

run_gan.py shows how to set the hyperparameters and network architectures. (multiple layers are supported).

The GAN automatically loads mnist images for a particular digit (default is 2). 

Credit to https://github.com/shinseung428/gan_numpy for his MNIST loading function. This work was utilised to help work through bugs regarding misshapen data.

### TODO: 
* Add support to change activation functions used when creating the GAN
* Test on larger datasets