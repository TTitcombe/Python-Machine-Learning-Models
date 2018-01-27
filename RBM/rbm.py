import numpy as np

class RBM(object):
    """An RBM Object"""
    
    def __init__(self,nInput, nHid, archs):
        self.nInput = nInput
        self.nHid = nHid
        self.archs = archs
        """Archs:
            learning rate
            batch size
            epochs
            k step
        """
        self.weights = np.random.randn((self.nInput,self.nHid))
        self.b = np.zeros((nInput))
        self.c = np.zeros((nHid))
        
    def train(self, x_train):
        bs = self.archs.batch_size
        batches = len(x_train[:,0]) / bs
        for batch in range(batches):
            dw = np.zeros((self.nInput,self.nHid))
            x_batch = x_train[batch*bs:(batch+1)*bs, :]
            for x in x_batch:
                posX = self.sample(x,None,'hidden')
                posX = posX * x
                
                #CD-1
                newVis = self.sample(x,posX, 'visible')
                negX = self.sample(newVis,None, 'hidden')
                negX = negX * newVis
                
                dw += (posX - negX)
            dw = dw / bs
            self.weights = self.weights - self.lr * dw
            
    def generate(self):
        newInputs = np.random.randint(0,2,self.nVis)
        for i in range(100):
            newHid = self.sample(newInputs,None,'hidden')
            newInputs = self.sample(None,newHid, 'visible')
        return newInputs
                
    def sample(self,inp,hid, hidden):
        if hidden == 'hidden':
            newHid = np.matmul(self.weights.T,inp)
            newHid = self.prob(inp, newHid, 100.0,'hidden')
            newHid[newHid >= 0.5] = 1
            newHid[newHid < 0.5] = 0
        elif hidden == 'visible':
            newHid = np.matmul(self.weights,hid)
            newHid = self.prob(hid, newHid, 100.0,'visible')
            newHid[newHid >= 0.5] = 1
            newHid[newHid < 0.5] = 0
        return newHid
                
                
    def h(self,nodes, hid):
        if hid == 'hidden':
            return np.matmul(self.weights.T,nodes) + self.c
        else:
            return np.matmul(self.weights,nodes) + self.b
                
                
    def prob(self,inp,hid,T,hidden):
        if hidden == 'hidden':
            exp = np.exp(-self.h(inp,hidden) * hid / T)
            return 1/(1+exp)
        else:
            exp = np.exp(-self.h(hid,hidden) * inp / T)
            return 1/(1+exp)