# Script of application of LSTM
from model import MyLSTM
import torch
import numpy as np
import matplotlib.pyplot as plt

# generate data functions
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

# generate data
num_samples = 200
X_seq, Y_seq = generate_data(num_samples=num_samples, method = sine)
print(X_seq.size(), Y_seq.size())

# split the data into training and testing sets
split_index = int(num_samples*0.8)
X_train = X_seq[:, :split_index, :]
Y_train = Y_seq[:, :split_index, :]
X_test = X_seq[:, split_index:, :]
Y_test = Y_seq[:, split_index:, :]

# initialize the model
model = MyLSTM(input_size=1, hidden_size=64, output_size=1)

# train the model
model.fit(X_train, Y_train, lr=1e-3, epochs=300)

# evaluate the model
print(model.evaluate(X_test, Y_test))

# plot the results
y_true = Y_test[:,:,:].squeeze().numpy()
y_pred = model(X_test)[0].detach().squeeze().numpy()
print(y_pred.shape, y_true.shape)
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('LSTM Prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()