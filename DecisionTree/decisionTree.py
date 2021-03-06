import numpy as np
from tree import tree

class DecisionTreeClassifier(object):
    
    def __init__(self):
        self.treeList = None #list of trees trained using one-versus-rest
        
    def __str__(self):
        s = "A Decision Tree classifier. \n"
        s += "Use train(x_train, list_of_attributes, y_train) to create a list of trained trees using one-versus-rest \n"
        s += "Use predict(x_test) to create predicted attributes for a given test set. \n"
        s += "Use getTrainedTrees() to return the list of trained one-versus-rest trees. \n"
        return s
        
    ##---------LEARNING FUNCTIONS----------------##
    def train_single_tree(self,examples, attributes, binary_targets):
        '''Creates a tree to split the given data.
        Inputs:
                    examples | n x F numpy array, binary (1 or 0), x data
            attributes | 1 x F numpy array, int, 'name' of attributes in data
            binary_targets | 1 x n numpy array, binary, target y value
        Outputs:
            a_tree | complete tree object'''
        if len(np.unique(binary_targets)) == 1:
            return tree(op=None, cls = binary_targets[0])
        elif sum(attributes > -1) == 0:
            return tree(op=None, cls=self.majority_value(binary_targets))
        else:
            best_attribute = self.choose_best_decision_attributes(examples, attributes, binary_targets)
            a_tree = tree(op=best_attribute)
            best_attribute_column = examples[:,best_attribute]
            #print sum(attributes > -1)
            #print best_attribute_column
            for possible_value in [0,1]:
                #each attribute can only be 0 or 1 (off or on)
                best_attribute_indices = best_attribute_column == possible_value
                #print best_attribute_indices
                new_examples = examples[best_attribute_indices,:]
    
                if not new_examples.size:
                    #if new_examples is empty
                    return tree(cls=self.majority_value(binary_targets))
    
                new_binary_targets = binary_targets[best_attribute_indices]
                new_attributes = attributes
                new_attributes[best_attribute] = -1
                subtree = self.train_single_tree(new_examples,new_attributes,new_binary_targets)
                a_tree.kids[possible_value] = subtree
        return a_tree

    def choose_best_decision_attributes(self,examples, attributes, binary_targets):
        '''Returns the attribute which splits the current tree the most
        Inputs:
            examples | as above
            attributes | as above
            binary_targets | as above
        Outputs:
            best_attribute | int, 'name' of attribute with highest information gain'''
        best_IG = -np.inf
        best_attribute = None
        for attribute in attributes:
            if attribute != -1:
                reduced_data = examples[:,attribute] #make an n x 1 numpy array
                IG = self.calc_gain(reduced_data, binary_targets)
                if IG > best_IG:
                    best_IG = IG
                    best_attribute = attribute
        return best_attribute

    def calc_gain(self,reduced_data, y):
        '''Calculates information gain for a particular attribute.
        Inputs:
            reduced_data | n x 1 np array, binary (1 or 0), x data for a single attribute
            y | 'binary_targets'
        Outputs:
            IG | float, information gain'''
        labels_are_1 = y == 1
        labels_are_0 = y == 0
        p = sum(labels_are_1)
        n = sum(labels_are_0)
        total_instances = float(p + n)
        IG = self.calc_entropy(p, n)
        for target_value in np.unique(reduced_data):
            pos, neg = self.count_pos(reduced_data, target_value, y)
            IG -= (pos+neg)*self.calc_entropy(pos, neg) / total_instances
        return IG
    
    def calc_entropy(self,pos, neg):
        '''Calculates entropy.
        Inputs:
            pos | int, number of positive examples found
            neg | int, number of negative exampels found
        Outputs:
            entropy | float'''
        if pos == 0 or neg == 0:
            return 0
        pos_frac = pos / float(pos+neg)
        neg_frac = neg / float(pos+neg)
        return -pos_frac*np.log2(pos_frac) - neg_frac*np.log2(neg_frac)

    def count_pos(self,reduced_data, target_value, y):
        '''Count number of data points where an attribute has value target_value,
            and the corresponding y value is 1 (positive counts) and 0 (negative counts).
        Inputs:
            reduced_data | n x 1 np array, (1 or 0), data for single attribute
            target_value | (1 or 0)
            y | 'binary_targets'
        Returns
            pos | int, number of positive examples
            neg | int, number of negative examples'''
        reduced_data_indices = (reduced_data == target_value)
        reduced_y = y[reduced_data_indices]
        #reduced_y = np.reshape(reduced_y, reduced_y.shape[0])
        pos = sum(reduced_y == 1)
        neg = sum(reduced_y == 0)
        return pos, neg

    def majority_value(self,binary_targets):
        '''Caculate mode of binary_targets'''
        counts = np.bincount(binary_targets, minlength=2) 
        if counts[0] >= counts[1]:
            return 0
        else: return 1

    def binary_labels(self,labels, label):
        '''Turns labels into binary for a particular value'''
        return (labels == label).astype(int)
    
    def train(self,examples, attributes, labels):
        '''For each value in labels, create a tree to test this. Add these trees to a list.
        Inputs:
            examples | n x F binary np array of training data
            labels | n x 1 np array, y values
        Outputs:
            tree_list | a list of trained trees, one for each value of labels.'''
        tree_list = []
        for label_value in np.unique(labels):
            a_binary_labels = self.binary_labels(labels, label_value)
            a_tree = self.train_single_tree(examples, attributes, a_binary_labels)
            tree_list.append(a_tree)
        self.treeList = tree_list
        return tree_list
    ##------------LEARNING FUNCTIONS END----------##
    

    ##------------EVALUATING A TREE---------------##
    def predict(self,x):
        '''Get prediction of each tree
        Inputs:
            t | list of tree objects
            x | 2-d numpy array of test data (n x F)
        Returns
            y | n x 1 numpy array of predicted target values'''
        if self.treeList == None:
            assert False, "You must train the model before predicting."
        all_data_predictions = np.zeros((x.shape[0],len(self.treeList)))
        predictions = np.zeros((x.shape[0]))
        for i, tree_object in enumerate(self.treeList):
            tree_prediction = self.predictBinary(tree_object, x)
            all_data_predictions[:,i] = tree_prediction
        for data_point in range(x.shape[0]):
            #loop through data points
            predicted_data = all_data_predictions[data_point,:]
            predicted_indices = np.argwhere(predicted_data == 1)
            if predicted_indices.size:
                predictions[data_point] = predicted_indices[0,0] + 1
            else:
                predictions[data_point] = 0
        return predictions

    def predictBinary(self,t, x):
        '''Predict binary for a tree
        Inputs:
            t | a tree object
            x | 2-d numpy array of test data (n x F)
        Returns
            y | n x 1 numpy array of predicted binary target value'''
        y = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            y[i] = self.traverse(t, x[i,:])
        return y
    
    def traverse(self,t, x):
        '''Traverse a tree for a data point to find the predicted binary target value.
        Inputs:
            t | a tree class
            x | 1 x F np array, binary, data for a particular example
        Returns:
            cls | 1 or 0, binary decision for particular y value'''
        #select attribute, traverse left or right
        print t.op, t.cls
        if t.op is None:
            return t.cls
        else:
            return self.traverse(t.kids[x[t.op]], x)
    ##----TREE EVALUATION FUNCTIONS END-------------##
    
    def getTrainedTrees(self):
        return self.treeList

