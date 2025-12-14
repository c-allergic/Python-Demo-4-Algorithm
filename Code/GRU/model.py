# GRU Model
import torch 
import torch.nn as nn
import torch.optim as optim

class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        
        # return a vector of size [batch_size, hidden_size]
        self.update_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        # return a vector of size [batch_size, hidden_size]
        self.reset_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.candidate_hidden_state = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
        
        self.h2o = nn.Linear(hidden_size, output_size)
        self.hidden_state = None
        
    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        hidden = torch.zeros(batch_size, self.hidden_size)
        outputs = []
        
        for i in range(seq_len):
            x_t = x[:, i, :]
            
            # gate update
            z_t = self.update_gate(torch.cat([hidden, x_t], dim=1))
            r_t = self.reset_gate(torch.cat([hidden, x_t], dim=1))
            
            # candidate hidden state update
            candidate_hidden = self.candidate_hidden_state(torch.cat([r_t * hidden, x_t], dim=1))
            
            # hidden state update
            hidden = (1 - z_t) * hidden + z_t * candidate_hidden
            output = self.h2o(hidden)
            outputs.append(output.unsqueeze(1))
        return torch.cat(outputs, dim=1), hidden
    
    def fit(self, x_train, y_train, epochs=100, lr=1e-3):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _ = self(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        return self
    
    def evaluate(self, x_test, y_test):
        criterion = nn.MSELoss()
        output, _ = self(x_test)
        loss = criterion(output, y_test)
        return loss.item()