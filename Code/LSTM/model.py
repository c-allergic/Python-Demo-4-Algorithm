# LSTM Model
import torch
import torch.nn as nn
import torch.optim as optim

class MyLSTM(nn.Module):
    """A single layer LSTM model.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        output_size (int): The size of the output features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MyLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        
        # LSTM has memory cell, input gate, forget gate, and output gate
        self.input_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.forget_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        
        self.output_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.candidate_cell_state = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

        self.h2o = nn.Linear(hidden_size, output_size)
        self.cell_state = None
        self.hidden_state = None
        
    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        hidden = torch.zeros(batch_size, self.hidden_size)
        cell_state = torch.zeros(batch_size, self.hidden_size)
        
        outputs = []
        
        for i in range(seq_len):
            x_t = x[:, i, :]
            # gate update
            f = self.forget_gate(torch.cat([x_t, hidden], dim=1))
            i = self.input_gate(torch.cat([x_t, hidden], dim=1))
            o = self.output_gate(torch.cat([x_t, hidden], dim=1))
            
            # cell state update
            candidate_cell_state = self.candidate_cell_state(torch.cat([x_t, hidden], dim=1))
            cell_state = f * cell_state + i * candidate_cell_state
            
            # hidden state update
            hidden = o * torch.tanh(cell_state)
            output = self.h2o(hidden)
            outputs.append(output.unsqueeze(1))
            
        self.cell_state = cell_state
        self.hidden_state = hidden
        return torch.cat(outputs, dim=1), self.hidden_state, self.cell_state
    
    def fit(self, x_train, y_train, epochs=100, lr=1e-3):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output, _, _ = self(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        return self
    
    def evaluate(self, x_test, y_test):
        criterion = nn.MSELoss()
        output, _, _ = self(x_test)
        loss = criterion(output, y_test)
        return loss.item()
