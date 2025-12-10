#import "@preview/noteworthy:0.2.0": *

#show: noteworthy.with(
  paper-size: "a4",
  font: "New Computer Modern",
  language: "EN",
  title: "LSTM",
  author: "Sheldon",
  contact-details: "https://github.com/c-allergic", // Optional: Maybe a link to your website, or phone number
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

For the backward process, we also use BPTT algorithm to train the LSTM.