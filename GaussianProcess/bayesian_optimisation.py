import numpy as np

from gp import GP

class BayesianOptimiser():
    '''A Bayesian Optimiser object.
    Plug in any model, define the hyperparams to vary,
    sit back, and relax.'''

    def __init__(self, model,X_train, y_train, X_val, y_val,
                    hyperparams, acq_func='PI',
                    loops = 1, iter_per_loop=10):
        '''Initialise the Bayesian Optimiser.
        Inputs:
            model | a machine learning model to optimise
            X_train | np array, training x data
            y_train | np array, training y data
            X_val | np array, validation x data (for GP)
            y_val | np array, validation y data
            hyperparams | dict, keys are hyperparam names, vals
                            are lists containing upper and lower bounds
            acq_func | string. Acquisition function, how the GP chooses the
                                next hyperparams
            loops | int. How many random initialisation you want
            iter_per_loop | int. Number of hyperparameter selections per
                                optimisation cycle
        Outputs:
            None
        '''

        if acq_func.lower() == 'pi':
            #Probability of Improvement
            self.acq_func = 'PI'
        else:
            raise NotImplementedError("Probability of Improvement" \
             "is the only acquisition function currently implemented")

        self.model = model
        self.hyperparams = hyperparams
        self._hyperparam_int_map()

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val


        self.loops = loops
        self.iter_per_loop = iter_per_loop

        self.x_data = []
        self.score_data = []

        self.best_score = 10e8
        self.best_hyperparams = None


    def optimise(reuse=True):
        '''Run an optimisation loop to find better
        model hyperparams.
        Inputs:
            reuse | bool, True if you want to carry on from where you left off
        Outputs:
            best_hyperarams | list
        '''

        if reuse:
            x = self.x_data
            y = self.score_data
        else:
            x = []
            y = []

        for loop in range(self.loops):
            for iter in range(self.iter_per_loop):

                if x != []:
                    gp = GP(x, y, 'matern')
                    new_x = self._choose_next(gp) #choose hyperparams to try next
                    new_x_dict = self._array_to_hyperparams(new_x)
                else:
                    #randomly choose
                    pass

                #Train and predict
                try:
                    self.model.train(self.X_train, self.y_train, **new_x_dict)
                    output_pred = self.model.predict(self.X_val)

                    #make sure it's an Nx1 array
                    if len(output_pred.shape) == 1:
                        output_pred = np.reshape(output_pred, (output_pred.shape[0],1))

                except Exception as e:
                    raise e("Model did not have method `train` " \
                                        "or `predict`")

                new_score = self._score(output_pred)

                if new_score < self.best_score:
                    self.best_score = new_score
                    self.best_hyperparams = new_x_dict
                    print("Best score is {}".format(new_score))
                    print("\n")

                x.append(new_x)
                y.append(new_score)

        return self.best_hyperparams



    def _score(self, output_pred):
        '''Calc misclassification rate.
        Inputs:
            output_pred | array, predicted y vals
        Returns:
            misclass | float, the misclassification rate
        '''
        diffs = output_pred != self.y_val
        misclass = float(sum(diffs) / output_pred.shape[0])
        return misclass


    def _choose_next(self, gp):
        '''Given a Gaussian Process trained on previous hyperparams (x),
        and misclassification (y), select the next hyperparams to try.
        Inputs:
            gp | A gaussian process
        Outputs:
            new_x | dict of hyperparams
        '''

        pass

    def _hyperparam_int_map(self):
        '''Create a dictionary mapping each hyperparams to an integer,
        and a dictionary mapping the integer to a hyperparam

        This is necessary as GP takes a np array as input, but we need
        to assign a hyperparam name to each value for human readable output.
        '''

        hyper_to_int = {}
        int_to_hyper = {}
        for i, hyperparam in enumerate(self.hyperparams.keys()):
            hyper_to_int[hyperparam] = i
            int_to_hyper[i] = hyperparam

        self.hyper_to_int = hyper_to_int
        self.int_to_hyper = int_to_hyper

    def _array_to_hyperparams(self, an_arr):
        '''Given an Nx1 numpy array, N being number of hyperparams
        to optimise, return a dict of hyperparameter values
        '''

        hypers = {}
        for i in range(an_arr.shape[0]):
            hyper_name = self.int_to_hyper[i]
            hypers[hyper_name] = an_arr[i,:]

        return hypers

    def _hypers_to_array(self, hypers):
        '''Given a dict of hyperparameter values, return
        a numpy array (each hyperparams has a specified place)
        '''

        an_arr = np.zeros((len(hypers.keys()),1))
        for hyper_name, val in hypers.items():
            hyper_int = self.hyper_to_int[hyper_name]
            an_arr[hyper_int,:] = val

    return an_arr
