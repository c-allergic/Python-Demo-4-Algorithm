import numpy as np

class DifferentialPrivacy:
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def laplace_mechanism(self, query_result, sensitivity):
        # b = sensitivity / epsilon
        b = sensitivity / self.epsilon
        # generate Laplace noise
        noise = np.random.laplace(0, b, 1)[0]
        # return the result with noise
        return query_result + noise

    def gaussian_mechanism(self,query_result, sensitivity):
        # sigma = sensitivity / epsilon
        sigma = sensitivity / self.epsilon
        # generate Gaussian noise
        noise = np.random.normal(0, sigma, 1)[0]
        # return the result with noise
        return query_result + noise