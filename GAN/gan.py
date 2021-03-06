import numpy as np
import warnings
import cv2
import os

from utilities import load_mnist, show_samples, FID
from ann import ANN

class GAN(object):
	def __init__(self, D_hidden_dim,
	 				G_hidden_dim, z_dim, hyperparams = {},
					dataset = 'mnist', image_dim = None):

		self.dataset = dataset
		if dataset.lower() == 'mnist':
			image_dim = 28*28
			self.digit = hyperparams.get("digit", 2)
		elif dataset.lower() == 'celeba_bw':
			print("This basic GAN version probably will not converge. \
				See TTitcombe/GANmodels for more powerful versions \
				(in development)")
			image_dim = 178 * 218
		elif dataset is None and image_dim is None:
			raise RuntimeError("You must either define a recognised dataset \
								or define an input image dimension")
		else:
			raise NotImplementedError("The dataset you have selected \
			 							is not recognised")

		self.epochs = hyperparams.get("epochs", 100)
		self.batchSize = hyperparams.get("batchSize", 64)
		self.lr = hyperparams.get("lr", 0.001)
		self.decay = hyperparams.get("decay", 1.)
		self.epsilon = hyperparams.get("epsilon", 1e-7) #avoid overflow

		self.D = ANN(image_dim,D_hidden_dim, 1, self.lr, False)
		self.G = ANN(z_dim,G_hidden_dim,image_dim,self.lr, True)

	def train(self, X_train = None):

		if self.dataset.lower() == 'mnist':
			X_train, N_train = load_mnist(self.digit)
			np.random.shuffle(X_train)
		elif self.dataset.lower() == 'celeba_bw':
			#X_train is a path to the images
			_, _, filenames = os.walk(X_train)
			N_train = len(filenames)
		elif X_train is None:
			raise RuntimeError("X training data must be provided")
		else:
			N_train = X_train.shape[0]
			np.random.shuffle(X_train)


		N_batch = N_train//self.batchSize
		for epoch in range(self.epochs):
			g_loss_tracker = [0.]
			g_loss_differences = []
			d_loss_tracker = []

			for step in range(N_batch):

				if self.dataset.lower() == 'celeba_bw':
					file = filenames[step]
					path = X_train + file
					X_batch = cv2.imread(path)
					X_batch = cv2.cvtColor( X_batch, cv2.COLOR_RGB2GRAY )
				else:
					X_batch = X_train[step*self.batchSize:(1+step)*self.batchSize]
					if X_batch.shape[0] != self.batchSize:
						break

				#Generate random (normal) z
				z = np.random.normal(loc=0.0,scale=0.5, size=(self.batchSize,100))
				z[z< -1] = -1.
				z[z > 1] = 1.

				#Feedforward
				g_logits, fake_img = self.G._feedforward(z)

				d_real_logits, d_real_output = self.D._feedforward(X_batch)
				d_fake_logits, d_fake_output = self.D._feedforward(fake_img, True)

				d_loss = -np.log(d_real_output+self.epsilon) - np.log(1 - d_fake_output+self.epsilon)

				#track D loss: Failure if 0; varying wildly is probably bad.
				assert np.mean(d_loss) > 1e-8, "D loss has gone to zero - Failure case"
				d_loss_tracker.append(d_loss)
				if step > 9:
					d_loss_tracker.pop(0)
					if np.std(d_loss_tracker) > 5: #what is "varying wildly"
						warnings.warn("D loss is varying sharply", UserWarning)

				g_loss = -np.log(d_fake_output+self.epsilon)

				#track G loss: "if it steadily decreases, it's fooling D with garbage"
				g_loss_differences.append(np.mean(g_loss - g_loss_tracker[-1]))
				if step > 8:
					g_loss_tracker.pop(0)
					if np.std(g_loss_differences) < 1e-2:
						warnings.warn("G loss is decreasing steadily. Check G output.", UserWarning)
				g_loss_tracker.append(np.mean(g_loss))

				#Update with decayed learning rate
				self.G.setLR(self.lr)
				self.D.setLR(self.lr)

				#Backprop
				d_archs = self.D.archs
				d_n_layers = self.D.N_layers
				d_lin_store = self.D.fake_lin_store
				d_act_store = self.D.fake_act_store
				self.D.backprop()
				self.G.backprop(d_n_layers, d_act_store, d_lin_store, d_archs)

				#Show samples
				samples = self.sample()
				full_image = show_samples(samples, 25, self.dataset)
				cv2.imshow('Samples', full_image)
				cv2.waitKey(1)

				#fid = FID(samples, X_train_reshaped)

				print("Epoch: %d; Step: %d; G Loss: %.4f; D Loss: %.4f; Real ac: %.4f; Fake ac: %.4f"%(epoch, step, np.mean(g_loss), np.mean(d_loss),np.mean(d_real_output), np.mean(d_fake_output)))



			self.lr = self.lr * self.decay
	def sample(self):
		z = np.random.normal(loc=0.0,scale=0.5, size=(self.batchSize,100))
		z[z< -1] = -1.
		z[z > 1] = 1.

		_, fake_img = self.G._feedforward(z)
		return fake_img
