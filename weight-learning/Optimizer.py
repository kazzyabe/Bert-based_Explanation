from scipy.special import expit
from numpy import dot

from CBR_Model import CBR_Model


class Optimizer(object):
    """ Optimizer

    A class used to optimize the weight, which provide static methods.

    TODO:
        * 
    """

    def __init__(self):
        pass
    
    @staticmethod
    def gradient_descent(model, obs, alpha=1e-2):
        """ Gradient Descent

        Gradient descent is a first-order iterative optimization algorithm for 
        finding the minimum of a function. To find a local minimum of a 
        function using gradient descent, one takes steps proportional to the 
        negative of the gradient (or approximate gradient) of the function at 
        the current point.
        https://www.pyimagesearch.com/2016/10/10/gradient-descent-with-python/

        Args:
            model (CBR_Model): Case-Based Reasoning Model
                The model which contains weight W to be updated. 

            obs (numpy.ndarray): observation matrix (instances)
                The shape of observation matrix should match the model.

            alpha (float): learning rate
                Also know as step size. Minimize the error when it is positive.

        Returns:
            A new weight after running gradient descent.
        """
        X = obs[:, :-1]
        y = obs[:, -1].reshape(obs.shape[0], -1)

        p = expit( dot(X, model.W) )    # predicted value
        e = y - p                       # error

        g = dot(X.T, e) / X.shape[0]    # gradient
        return model.W - alpha * g
