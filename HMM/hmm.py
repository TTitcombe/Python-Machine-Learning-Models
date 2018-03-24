"""Object for a Hidden Markov Model.
This was created as part of coursework for the Imperial College London
course 'Advanced Statistical Machine Learning'.

Code wrapper created by Stefanos Zafeiriou"""
import numpy as np
from collections import namedtuple
import scipy.stats

class hmm(object):

    def __init__(self, Y, model, init, Nhidden, Niter, epsilon, precision = 1e-9):
        self.Y = Y
        self.T = Y.shape[1]
        self.init = init
        self.Nhidden = Nhidden
        self.Niter = Niter
        self.epsilon
        self.precision = precision
        self.model = model

        InitGaussian = namedtuple('InitGaussian', ['A', 'Means', 'Variances', 'pi'])
        InitMultinomial = namedtuple('InitMultinomial', ['A', 'B', 'pi'])

        if model == "gauss":
            self.EM = self.EM_estimate_gauss
            A, Mu, Sigma, Pi = self.EM()
            smallB = lambda X : self.computeSmallB_Gaussian(X, Mu, Sigma)
            self.params = InitGaussian(A, Mu, Sigma, Pi)
        elif model == "multinomial":
            self.EM = self.EM_estimate_multinomial
            A, B, Pi = self.EM()
            smallB = lambda X : self.computeSmallB_Discrete(X, B)
            self.params = InitMultinomial(A, B, Pi)
        else:
            raise ValueError('Model must be gauss or multinomial')

        self.latentPath = self.ViterbiDecode(smallB, A, Pi)

    def getLatentPath(self):
        return self.latentPath

    def getParams(self):
        return self.params

    def calcAccuracy(self, actualLatenPath):
        accuracy = (actualLatentPath == self.latentPath).sum() / float(self.latentPath.size)
        return accuracy

    def normalise(self, A, dim=None):
        precision = self.precision

        if dim is not None and dim > 1:
            raise ValueError("Normalise doesn't support more than 2 dimensions")

        z = A.sum(dim)
        if z.shape:
            z[np.abs(z) < precision] = 1
        elif np.abs(z) < precision:
            return 0,1

        if dim == 1:
            return np.transpose(A.T / z), z
        else:
            return A / z, z

        def computeSmallB_Gaussian(self, Y, Means, Variances):
            """Compute probabilities for the data points Y for a Gaussian model.

            Inputs:
                - Y: the data - 1 x T
                - Means: vector of current mean estimates - D x K
                - Variances: vector of current variance estimates
                - Nhidden: aka K, number of hidden states
                - T: length of the sequence
            Output:
                - b: vector of observation probabilities """
            Nhidden = self.Nhidden
            T = self.T
            b = np.zeros((Nhidden, T))
            for t in range(T):
                for n in range(Nhidden):
                    b[n,t] = scipy.stats.norm.pdf(Y[t], loc=Means[n,0], scale=Variances[n,0])

            return b

        def computeSmallB_Discrete(self, Y, B):
            """Compute probabilities for the data points Y for a discrete model.

            Inputs:
                - Y: the data - 1 x T (not onehot)
                - B: observation probabilities - Nhidden x Nobs
            Output:
                - b: vector of observation probabilities """
            b = np.zeros((B.shape[0], Y.shape[0]))
            for t in range(Y.shape[0]):
                b[:,t] = B[:,Y[t]-1]

            return b

        def BackwardFiltering(self, A, b):
            N = self.Nhidden
            T = self.T
            beta = np.zeros((N,T))
            beta[:,-1] = 1.0 #initialise last elements
            for t in range(1,T):
                terms1 = b[:,t] * beta[:,-t]
                prev_beta = np.dot(A, terms1)
                beta[:,-(t+1)], _ = self.normalise(prev_beta)
            return beta

        def ForwardFiltering(self, A, b, pi):
            N = self.Nhidden
            T = self.T
            alpha = np.zeros((N, T))
            Z = np.zeros((T))

            alpha[:,0], Z[0] = self.normalise(pi[:,0] * b[:,0])
            for t in range(1,T):
                alpha[:,t], Z[t] = self.normalise(b[:,t] * np.dot(A.T, alpha[:,t-1]))

            logProb = sum(np.log(Z))
            return alpha, logProb, Z

        def ForwardBackwardSmoothing(self, A, b, pi):
            N = self.Nhidden
            T = self.T
            alpha, logProb, Z = self.ForwardFiltering(A, b, pi, N, T)
            beta = self.BackwardFiltering(A, b, N, T)
            gamma = alpha * beta
            gamma_norm, _ = self.normalise(gamma, dim=0)

            return alpha, beta, gamma_norm, logProb, Z

        def SmoothedMarginals(self, A, b, alpha, beta):
            NHidden = self.Nhidden
            T = self.T
            marginal = np.zeros((Nhidden, NHidden, T-1))

            for t in range(T-1):
                normed, _ = self.normalise(A * np.dot(alpha[:,t], np.transpose((b[:,t+1] * beta[:,t+1]))))
                marginal[:,:,t] = normed
            return marginal

        def EM_estimate_gauss(self):
            Y = self.Y
            Nhidden = self.Nhidden
            Niter = self.Niter
            epsilon = self.epsilon
            init = self.init

            N, T = Y.shape
            A = init.A
            old_A = np.ones(A.shape)

            Means = init.Means
            Variances = init.Variances

            pi = init.pi
            old_pi = np.zeros(pi.shape)

            i = 0
            A_diff = abs(old_A - A).any() < epsilon
            pi_diff = abs(old_pi - pi).any() < epsilon

            while i < Niter and not A_diff and not pi_diff:
                new_pi = np.zeros(pi.shape)
                new_A = np.zeros(A.shape)
                new_means = np.zeros(Means.shape)
                new_means_denominator = np.zeros(Means.shape)
                new_variances = np.zeros(Variances.shape)
                new_variances_denominator = np.zeros(Variances.shape)

                old_logprob = 0.0
                for l in range(N):
                    data = Y[l,:]
                    b = self.computeSmallB_Gaussian(data, Means, Variances, Nhidden, T)
                    alpha, beta, gamma, logProb, Z = self.ForwardBackwardSmoothing(A, b, pi, Nhidden, T)
                    if abs(old_logprob - logProb) < epsilon:
                        print("LogProb not changing")
                        break
                    old_logprob = logProb
                    marginal = self.SmoothedMarginals(A, b, alpha, beta, T, Nhidden)

                    new_pi[:,0] += gamma[:,0]
                    for j in range(A.shape[0]):
                        for k in range(A.shape[1]):
                            new_A[j,k] += sum(marginal[j,k,:])
                    new_means[:,0] += np.dot(gamma, data.T)
                    new_means_denominator[:,0] += np.sum(gamma, axis=1) #sum along t

                    for t in range(T):
                        for k in range(Nhidden):
                            new_variances[k,:] += gamma[k,t] * np.dot((data[t] - Means[k,0]),(data[t] - Means[k,0]).T)
                    new_variances_denominator[:,0] += np.sum(gamma, axis=1)


                Means = new_means / new_means_denominator
                Variances = new_variances / new_variances_denominator

                old_pi = pi
                pi, _ = normalize(new_pi, dim=0)
                pi_diff = abs(old_pi - pi).any() < epsilon

                old_A = A
                A, _ = normalize(new_A, dim=1)
                A_diff = abs(old_A - A).any() < epsilon

                i += 1

            return A, Means, Variances, pi

        def EM_estimate_multinomial(self):
            Y = self.Y
            Nhidden = self.Nhidden
            Niter = self.Niter
            epsilon = self.epsilon
            init = self.init

            N, T = Y.shape

            A = init.A
            old_A = np.zeros(A.shape)
            B = init.B
            old_B = np.zeros(B.shape)
            pi = init.pi
            old_pi = np.zeros(pi.shape)

            i = 0

            A_diff = abs(old_A - A).any() < epsilon
            pi_diff = abs(old_pi - pi).any() < epsilon

            while i<Niter and not A_diff and not pi_diff:
                new_pi = np.zeros(pi.shape)
                new_A = np.zeros(A.shape)
                new_B = np.zeros(B.shape)
                new_B_denominator = np.zeros(B.shape)

                old_logProb = 0.0

                for l in range(N):
                    data = Y[l,:]
                    b = self.computeSmallB_Discrete(data, B)

                    alpha, beta, gamma, logProb, Z = self.ForwardBackwardSmoothing(A, b, pi, Nhidden, T)
                    if abs(old_logProb - logProb) < epsilon:
                        print("logProb not changing")
                        break
                    marginal = self.SmoothedMarginals(A, b, alpha, beta, T, Nhidden)


                    new_pi[:,0] += gamma[:,0]

                    for j in range(A.shape[0]):
                        for k in range(A.shape[1]):
                            new_A[j,k] += sum(marginal[j,k,:])


                    Nv = len(np.unique(data))
                    X = np.zeros((Nv, T))
                    for t in range(T):
                        X[data[t]-1,t] = 1
                    for j in range(B.shape[0]):
                        for k in range(B.shape[1]):
                            new_B[j,k] += sum(gamma[j,:] * X[j,:])
                            new_B_denominator[j,k] += sum(gamma[j,:])

                new_B = new_B / new_B_denominator

                old_pi = pi
                pi, _ = self.normalise(new_pi, dim=0)
                pi_diff = abs(old_pi - pi).any() < epsilon

                old_B = B
                B, _ = self.normalise(new_B, dim=1)
                B_diff = abs(old_B - B).any() < epsilon

                old_A = A
                A, _ = self.normalise(new_A, dim=1)
                A_diff = abs(old_A - A).any() <epsilon

                i += 1

            return A, B, pi

        def ViterbiDecode(self, smallB, A, Pi):
            Y = self.Y
            S = np.zeros(Y.shape)

            log_A = np.log2(A)
            log_pi = np.log2(Pi)

            for seq in range(Y.shape[0]):
                delta_this_sequence = np.zeros((Nhidden, Y.shape[1])) #path probabilities for this sequence

                data = Y[seq,:]
                log_data = np.log2(data)

                b = smallB(data)
                log_b = np.log2(b)

                delta_this_sequence[:,0] = log_pi[:,0] + log_b[:,0]
                for t in range(1, Y.shape[1]):
                    for n in range(Nhidden):
                        delta_this_sequence[n,t] = log_b[n,t] + np.max(delta_this_sequence[:,t-1] + log_A[n,:]) #way to do this with matrix mult?
                predicted_states = np.argmax(delta_this_sequence, axis=0) + 1 #plus 1 as states begin at 1
                S[seq,:] = predicted_states
            return S.astype(int)
