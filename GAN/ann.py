import numpy as np

class ANN(object):
    '''An artificial neural network object, will form basis of D and G nets.
    Currently only supports 1 hidden layer.'''

    def __init__(self, input_dim, hidden_dims,output_dim, lr, G, seed=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dims
        self.N_layers = 1 + len(hidden_dims)
        self.output_dim = output_dim
        self.seed = seed
        if seed:
            np.random.seed(seed)
        self.lr = lr
        self.batchSize = None
        self.epsilon = 10e-8
        self.randomInit()
        self.G = G

        if G:
            self.final_act = self.tanh
            self.final_back_act = self.back_tanh
            self.backprop = self.G_backprop
        else:
            self.final_act = self.sigmoid
            self.final_back_act = self.back_sigmoid
            self.backprop = self.D_backprop

    def randomInit(self):
            archs = {}

            archs['W0'] = np.random.randn(self.input_dim,self.hidden_dim[0]).astype(np.float32) * np.sqrt(2.0/(self.input_dim))
            archs['b0'] = np.zeros(self.hidden_dim[0]).astype(np.float32)

            for i in range(1,self.N_layers-1):
                W = np.random.randn(self.hidden_dim[i-1], self.hidden_dim[i]).astype(np.float32) * np.sqrt(2.0/(self.hidden_dim[i-1]))
                b = np.zeros(self.hidden_dim[i]).astype(np.float32)
                archs['W{}'.format(i)] = W
                archs['b{}'.format(i)] = b
            archs['W{}'.format(self.N_layers-1)] = np.random.randn(self.hidden_dim[-1],self.output_dim).astype(np.float32) * np.sqrt(2.0/(self.hidden_dim[-1]))
            archs['b{}'.format(self.N_layers-1)] = np.zeros(self.output_dim).astype(np.float32)
            self.archs = archs

    def setLR(self,newLR):
        self.lr = newLR

    def linear(self,X, W, b):
        return np.dot(X,W) + b

    def back_linear(self, X, W, b, dE):
        dW = np.dot(X.T,dE)
        dX = np.dot(dE, W.T)
        db = np.sum(dE,axis=0)
        return dX, dW, db

    def sigmoid(self,input):
        return 1 / (1 + np.exp(-input))

    def back_sigmoid(self,input):
        o = self.sigmoid(input)
        return o * (1 - o)

    def relu(self,input):
        x = input.copy()
        x[ x < 0] = 0.
        return x

    def back_relu(self,input):
        x = np.ones(input.shape).astype(np.float32)
        x[input < 0] = 0.
        return x

    def lrelu(self,input, alpha=0.01):
        x = input.copy()
        x[input < 0] = alpha * x[input < 0]
        return x

    def back_lrelu(self,input, alpha = 0.01):
        x = np.ones(input.shape).astype(np.float32)
        x[input < 0] = alpha
        return x

    def tanh(self,input):
        return np.tanh(input)

    def back_tanh(self,input):
        t = self.tanh(input)
        return 1. - t**2

    def feedforward(self, input, fake=False):
        lin_store = {}
        act_store = {}

        if self.batchSize is None:
            self.batchSize = input.shape[0]

        act_store['0'] = np.reshape(input, (self.batchSize,-1))

        for i in range(1,self.N_layers):
            lin_store[str(i)] = act_store[str(i-1)].dot(self.archs['W{}'.format(i-1)]) + self.archs['b{}'.format(i-1)]
            act_store[str(i)] = self.lrelu(lin_store[str(i)])

        lin_store[str(self.N_layers)] = act_store[str(self.N_layers-1)].dot(self.archs['W{}'.format(self.N_layers-1)]) + self.archs['b{}'.format(self.N_layers-1)]
        act_store[str(self.N_layers)] = self.final_act(lin_store[str(self.N_layers)])

        if self.G:
            act_store[str(self.N_layers)] = np.reshape(act_store[str(self.N_layers)], (self.batchSize,28,28))

        if fake:
            self.fake_lin_store = lin_store
            self.fake_act_store = act_store
        else:
            self.lin_store = lin_store
            self.act_store = act_store

        return lin_store[str(self.N_layers)], act_store[str(self.N_layers)]


    def G_backprop(self,D_N_layers, D_act_store, D_lin_store,
                    D_archs):

        #fake_input = np.reshape(fake_input, (self.batchSize,-1))
        fake_input = self.act_store[str(self.N_layers)] #input into the Discriminator


        g_loss = np.reshape(D_act_store[str(D_N_layers)], (self.batchSize, -1))
        g_loss = -1.0/(g_loss + self.epsilon)

        #Pass error through descriminator
        loss_deriv = g_loss*self.back_sigmoid(D_lin_store[str(D_N_layers)])
        for j in range(D_N_layers-1,-1,-1):
            loss_deriv = loss_deriv.dot(D_archs['W{}'.format(j)].T)
            if j > 0:
                loss_deriv = loss_deriv*self.back_lrelu(D_lin_store[str(j)])


		#Pass error back through Generator
        loss_deriv = loss_deriv*self.back_tanh(self.lin_store[str(self.N_layers)])

        for i in range(self.N_layers-1,-1,-1):
            prev_layer = np.expand_dims(self.act_store[str(i)], axis=-1)
            loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
            grad_W = np.matmul(prev_layer,loss_deriv_)
            grad_b = loss_deriv
            for idx in range(self.batchSize):
                self.archs['W{}'.format(i)] = self.archs['W{}'.format(i)] - self.lr * grad_W[idx]
                self.archs['b{}'.format(i)] = self.archs['b{}'.format(i)] - self.lr * grad_b[idx]

            if i > 0:
                loss_deriv = loss_deriv.dot(self.archs['W{}'.format(i)].T)
                loss_deriv = loss_deriv*self.back_lrelu(self.lin_store[str(i)])


    def D_backprop(self):

        d_real_loss = -1.0/(self.act_store[str(self.N_layers)] + self.epsilon)
        d_fake_loss = -1.0/(self.fake_act_store[str(self.N_layers)] - 1.0 + self.epsilon)


        loss_deriv = d_real_loss*self.back_sigmoid(self.lin_store[str(self.N_layers)])
        loss_deriv_fake = d_fake_loss*self.back_sigmoid(self.fake_lin_store[str(self.N_layers)])

        new_archs = self.archs.copy() #Create store of weights and biases
        for j in range(self.N_layers-1, -1, -1):
            real_input = self.act_store[str(j)]
            fake_input = self.fake_act_store[str(j)]
            if j == 0:
                real_input = np.reshape(real_input, (self.batchSize,-1))
                fake_input = np.reshape(fake_input, (self.batchSize,-1))

            prev_layer = np.expand_dims(real_input, axis=-1)
            prev_layer_fake = np.expand_dims(fake_input, axis=-1)

            loss_deriv_ = np.expand_dims(loss_deriv, axis=1)
            loss_deriv_fake_ = np.expand_dims(loss_deriv_fake, axis=1)

            grad_W_real =  np.matmul(prev_layer,loss_deriv_)
            grad_W_fake = np.matmul(prev_layer_fake,loss_deriv_fake_)

            grad_b_real = loss_deriv
            grad_b_fake = loss_deriv_fake

            grad_W = grad_W_real + grad_W_fake
            grad_b = grad_b_real + grad_b_fake

            #Update new archs (propagate error using old archs)
            for idx in range(self.batchSize):
                new_archs['W{}'.format(j)] = new_archs['W{}'.format(j)] - self.lr*grad_W[idx]
                new_archs['b{}'.format(j)] = new_archs['b{}'.format(j)] - self.lr*grad_b[idx]

            if j > 0:
                loss_deriv = loss_deriv.dot(self.archs['W{}'.format(j)].T)
                loss_deriv_fake = loss_deriv_fake.dot(self.archs['W{}'.format(j)].T)

                loss_deriv = loss_deriv*self.back_lrelu(self.lin_store[str(j)])
                loss_deriv_fake = loss_deriv_fake*self.back_lrelu(self.fake_lin_store[str(j)])

        self.archs = new_archs
