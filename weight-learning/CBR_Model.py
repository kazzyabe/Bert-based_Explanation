from numpy.random import uniform as random_uniform
from numpy import array
from numpy import dot
from numpy import mean
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score


class CBR_Model(object):
    """ Case-Based Reasoning Model

    This module is a Case-Based Reasoning Model contains the inforamtion needed 
    to do prediction. Also, a prediction method is provided. 

    Attributes:
        N (int):
            N features for each case.

        W (numpy.ndarray): 
            weight matrix with shape (N, 1)
        
        diff_fns (function array):
            An array of N local similarity functions. Each function takes 2 input 
            arguments and returns a positive float number for their difference.
            Return 0 means two arguments are same.
    
    TODO:
        * Check the shape of arguments.
    """

    def __init__(self, diff_fns=[]):
        """ CBR_Model Initialization

        The initialization method for CBR_Model. Record the total number of
        features and the difference functions for each feature in given order. 
        Also, a weight will be generated randomly.

        Args:
            diff_fns (function array):
                A array of N local similarity functions. Each function takes 2 input 
                arguments and returns a positive float number for their 
                difference. Return 1 means two arguments are same.
        """
        self.N = len(diff_fns)
        self.W = random_uniform(size=(self.N, 1))
        self.diff_fns = diff_fns
    
    def _similarity(self, first, second):
        """ Get similarity for given cases

        The sum of similarity for all features. The similarity for given 
        feature is compute by diff_func(first[feat], second[feat]) * W[feat].

        Args:
            first, second (numpy.ndarray): Cases used to compute similarity. 
                A matrix with 1 row and N cols.
        
        Returns:
            A positive float number. 1 for same. 
        """
        diff_mat = array([
            diff(first.item(i), second.item(i)) 
            for i, diff in enumerate(self.diff_fns)
        ])
        diff_score = dot(diff_mat, self.W).item(0)
        # import pdb
        # pdb.set_trace()
        return diff_score

    def predict(self, case, retrv, k=1):
        """ Prediction Method

        Use retrived cases as knowledge base, and do prediction for new case.
        The prediction takes k most similar cases and compute their average.

        Args:
            case (numpy.ndarray): New case
                A matrix with 1 row and N cols.

            retrv (numpy.ndarray): Retrieved cases
                A matrix with N+1 cols. The last col is the Y value.
            
            k (int): 
                Use k less difference cases.
        
        Returns:
            The predicted value.
        """
        diff_scores = array([
            self._similarity(case, retrv_i[:-1]) for retrv_i in retrv
        ])
        k_index = diff_scores.argsort()[:k]
        p_value = mean(retrv[k_index][:, -1])
        return p_value

    def test_acc(self, cases, k=1, act_fn=None):
        """ Test Accuracy for Given Cases 

        This method will take cases, split them with leave one out algorithm. 
        Use the one as test case and others as retrieved cases to do prediction
        and then return the percentage of correct prediction. 

        Args:
            cases (numpy.ndarray): 
                A matrix with N+1 cols. The last col is the Y value.
            
            k (int):
                k has to be an odd number so we don't have to insert any biases.
                Use k less difference cases for prediction.

            act_fn (function):
                This function will be called after call predict(). It can be 
                used to convert the predicted result to same format as Y.
        
        Returns:
            The percentage of correct prediction. From 0 to 1 (included).
        """
        correct_count = 0
        res = {'P': [], 'T':[]}
        for train_index, test_index in LeaveOneOut().split(cases):
            train, test = cases[train_index], cases[test_index]
            predict = self.predict(test[0][:-1], train, k=k)      # predict define above in row 67
            p = act_fn(predict) if act_fn is not None else predict
            correct_count += 1 if p == test[0][-1] else 0
            res['P'].append(p)
            res['T'].append(test[0][-1])
        accuracy = correct_count / cases.shape[0]
        f1 = f1_score(res['T'], res['P'])
        return accuracy, f1
