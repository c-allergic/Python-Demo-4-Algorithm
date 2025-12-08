# script for synthetic data generation
import numpy as np
import torch

def sine(x):
    return np.sin(x)

def cosine(x):
    return np.cos(x)

# generate a sine wave
def generate_data(num_samples=200, method = None):
    # numerically generate the data
    x = np.linspace(0, 10*np.pi, num_samples)
    y = method(x)
    
    # add noise to the data
    np.random.seed(42)
    y = y + np.random.normal(0, 0.01, num_samples)

    # format the data into tensor of shape (batch_size, seq_len, input_size)
    y_tensor = torch.from_numpy(y).float().reshape(1, num_samples, 1)
    
    X_seq = y_tensor[:,:-1,:]
    Y_seq = y_tensor[:,1:,:]
    
    return X_seq, Y_seq