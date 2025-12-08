# A python implementation of Recurrent Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MyRNN(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(MyRNN, self).__init__()

        self.hidden_size = hidden_size
        
        # input layer
        self.i2h = nn.Linear(input_size, hidden_size)
        
        # hidden layer
        self.h2h = nn.Linear(hidden_size, hidden_size)
        
        # output layer
        self.h2o = nn.Linear(hidden_size, output_size)
        
        # activation function
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        """
        Args:
            x (tensor): input tensor, shape: (batch_size, seq_len, input_size)
        
        Returns:
            output (tensor): output tensor, shape: (batch_size, output_size)
        """
        batch_size, seq_len, input_size = x.size()
        
        # initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        
        for i in range(seq_len):
           x_t = x[:, i, :]
           combined = self.i2h(x_t) + self.h2h(hidden)
           hidden = self.tanh(combined)
           
           output = self.h2o(hidden)
           outputs.append(output.unsqueeze(1))
           
        return torch.cat(outputs, dim=1), hidden

    def fit(self, x_train, y_train, epochs=100, lr = 1e-3):
        # define the loss function
        criterion = nn.MSELoss()
        # define the optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # train the model
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = self(x_train)
            loss = criterion(output, y_train)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            loss.backward()
            optimizer.step()
        return self

    def evaluate(self, x_test, y_test):
        # define the loss function
        criterion = nn.MSELoss()
        # evaluate the model
        output, _ = self(x_test)
        loss = criterion(output, y_test)
        return loss.item()
    
    def rolling_predict(self, initial_input, steps=10):
        self.eval()
        
        current_input =  initial_input.clone()
        predictions = []
        
        with torch.no_grad():
            for _ in range(steps):
                # generate predicted sequence throught forward pass
                output, _ = self(current_input)
                last_prediction = output[:,-1,:].unsqueeze(1)
                
                # append the predicted sequence to the predictions list
                predictions.append(last_prediction)
                
                # update the current input
                current_input = torch.cat([current_input[:,1:,:], last_prediction], dim=1)
                
        return torch.cat(predictions, dim=1)
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, hidden = self.rnn(x)
        output = self.fc(output)
        return output, hidden
    
    def fit(self, x_train, y_train, epochs=100, lr = 1e-3):
        # define the loss function
        criterion = nn.MSELoss()
        # define the optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)
        # train the model
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = self(x_train)
            loss = criterion(output, y_train)
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            loss.backward()
            optimizer.step()
        return self
    
    def evaluate(self, x_test, y_test):
        # define the loss function
        criterion = nn.MSELoss()
        # evaluate the model
        output, _ = self(x_test)
        loss = criterion(output, y_test)
        return loss.item()