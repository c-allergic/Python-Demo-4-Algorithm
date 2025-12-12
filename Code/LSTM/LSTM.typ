#import "../noteworthy/lib.typ": *
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *

#show: codly-init.with()
#show: noteworthy.with(
  paper-size: "a4",
  font: "New Computer Modern",
  language: "EN",
  title: "LSTM",
  author: "Sheldon",
  contact-details: link("https://github.com/c-allergic",text[c-allergic]), // Optional: Maybe a link to your website, or phone number
  toc-title: "Table of Contents",
  // watermark: "DRAFT", // Optional: Watermark for the document
)

= LSTM
== Mathematical Formulation
When we train RNN, we usually face the problem of vanishing gradient or exploding gradient.

To solve this problem, we can use LSTM, whose architecture use simple addition and multiplication to update the memory.

LSTM introduces some new components compared to RNN:
#definition[
- Memory Cell $C_t$: The memory cell is a vector that stores the long-term memory. It updates at every time step.
- Gate: 
  - Input Gate $i_t$: Control what information should be added to the memory.
  - Forget Gate $f_t$: Control what information should be forgotten from the memory.
  - Output Gate $o_t$: Control what information should be output to the next memory cell $C_(t+1)$.
- Hidden State $h_t$: Just like RNN, LSTM has $h_t$, but the hidden state is a vector that stores the short-term memory. 
]

Before introducing the forward process, we need to introduce the hadamard product.
#definition[
  #linebreak()
  The hadamard product of two vectors $a in RR^n$ and $b in RR^n$ is defined as: $ a dot.o b = [a_1 b_1, a_2 b_2, ..., a_n b_n] $
]


For a LSTM of $[0,T]$ time steps, the whole forward process can be described as:
#definition[
#linebreak()
At timestep $t$:
+ Gate update: 
  - Forget Gate: $f_t = sigma(W_f dot [h_(t-1), x_t] + b_f)$
  
  - Input Gate: $i_t = sigma(W_i dot [h_(t-1), x_t] + b_i), tilde(C)_t = tanh(W_c dot [h_(t-1), x_t] + b_C)$

  - Output Gate: $o_t = sigma(W_o dot [h_(t-1), x_t] + b_o)$ 

+ Memory Cell update: $C_t = f_t dot.o C_(t-1) + i_t dot.o tilde(C)_t$ 

+ Hidden State update: $h_t = o_t dot tanh(C_t)$

+ (Optional) Output: $y_t = f(W_y h_t + b_y)$
]

Let's do a *dimension analysis*:
#example[
  #let hidden_size = $N_"hidden"$
  #let input_size = $N_"input"$
  
  For an input sequence $x$ of size $#input_size = D $, the dimension of different parts in the LSTM at timestep $t$ are:
  - Gates:
    - $sigma(RR^(#hidden_size times (#input_size + #hidden_size)) dot RR^((#hidden_size + #input_size) times 1)) -> f_t in RR^(#hidden_size times 1)$

    - Input Gate and Output Gate are similar  
    
  -  Memory Cell: $ C_t in [(RR^(#hidden_size times 1) dot.o RR^(#hidden_size times 1)) + (RR^(#hidden_size times 1) dot.o RR^(#hidden_size times 1))] -> C_t in RR^(#hidden_size times 1) $
]

For the backward process, we also use *BPTT algorithm* to train the LSTM.

= Code Implementation
Most of the code is similar to the RNN implementation, only the initialization of the model is different. Because we have different components.

```python
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
```

And the forward process follows the mathematical formulation:
```python
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
```

= Thinking
LSTM is a powerful model that can capture the long-term dependencies in the data. However, the composition of the model is quite complex and it introduces more parameters to train. 