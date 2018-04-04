import numpy as np

from load_mnist import load_mnist
from ann import ANN


class GAN(object):
	def __init__(self,image_dim, D_hidden_dim,
	 				G_hidden_dim, z_dim, output_dim = 1, hyperparams = {}):

		self.number = hyperparams.get("number", 2)
		self.epochs = hyperparams.get("epochs", 100)
		self.batchSize = hyperparams.get("batchSize", 64)
		self.lr = hyperparams.get("lr", 0.001)
		self.decay = hyperparams.get("decay", 1.)
		self.epsilon = hyperparams.get("epsilon", 1e-7) #avoid overflow

		self.D = ANN(image_dim,D_hidden_dim,output_dim, self.lr, False)
		self.G = ANN(z_dim,G_hidden_dim,image_dim,self.lr, True)

	def train(self):
		X_train, N_train = load_mnist(self.number)
		np.random.shuffle(X_train)

		N_batch = N_train//self.batchSize
		for epoch in range(self.epochs):
			for step in range(N_batch):

				X_batch = X_train[step*self.batchSize:(1+step)*self.batchSize]
				if X_batch.shape[0] != self.batchSize:
					break

				#Generate random (normal) z
				z = np.random.normal(loc=0.0,scale=0.5, size=(self.batchSize,100))
				z[z< -1] = -1.
				z[z > 1] = 1.

				#Feedforward
				g_logits, fake_img = self.G.feedforward(z)

				d_real_logits, d_real_output = self.D.feedforward(X_batch)
				d_fake_logits, d_fake_output = self.D.feedforward(fake_img, True)

				d_loss = -np.log(d_real_output+self.epsilon) - np.log(1 - d_fake_output+self.epsilon)

				g_loss = -np.log(d_fake_output+self.epsilon)

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

				print("Epoch: [%d]; Step: [%d]; G Loss: %.4f; D Loss: %.4f; Real acc: %.4f; Fake acc: %.4f"%(epoch, step, np.mean(g_loss), np.mean(d_loss), np.mean(d_real_output), np.mean(d_fake_output)))

			self.lr = self.lr * self.decay

	def sample(self, N):
		z = np.random.normal(loc=0.0,scale=0.5, size=(N,100))
		z[z< -1] = -1.
		z[z > 1] = 1.

		_, fake_img = self.G.feedforward(z)
		return fake_img