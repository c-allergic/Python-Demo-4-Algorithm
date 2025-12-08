# main script for the RNN
from rnn import MyRNN, RNN
from data import generate_data, sine, cosine
import matplotlib.pyplot as plt

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

# create the model
model = MyRNN(input_size=1, hidden_size=64, output_size=1)

# train the model
# model.fit(X_train, Y_train, lr=1e-3, epochs=300)
model.fit(X_seq, Y_seq, lr=1e-3, epochs=300)

# evaluate the model
print(model.evaluate(X_test, Y_test))

# plot the results
y_true = Y_seq[:,10:39,:].squeeze().numpy()
# y_pred = model(X_test)[0].detach().squeeze().numpy()
y_pred = model.rolling_predict(X_test[:,0:10,:], steps=29).detach().squeeze().numpy()
print(y_pred.shape, y_true.shape)
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.title('RNN Prediction')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
