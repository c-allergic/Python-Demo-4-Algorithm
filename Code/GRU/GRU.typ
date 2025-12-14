#import "../noteworthy/lib.typ": *

#show: noteworthy.with(
  paper-size: "a4",
  font: "New Computer Modern",
  language: "EN",
  title: "GRU",
  author: "Sheldon",
  contact-details: link("https://github.com/c-allergic",text[c-allergic]),
  toc-title: none,
)

// Write here
= GRU
== Mathematical Formulation
GRU is a variant of LSTM, which is more efficient and easier to train.

It has less gates compared to LSTM, which makes it more efficient.


+ Update Gate $z_t$: Control how much of the new candidate hidden state should be used to update the hidden state and how much of the previous hidden state should be kept.
  $
    z_t = sigma(W_z dot [h_(t-1), x_t] + b_z)
    $
+ Reset Gate $r_t$: Control how much of the previous hidden state should be used to update the candidate hidden state.
  $
    r_t = sigma(W_r dot [h_(t-1), x_t] + b_r)
    $
+ Candidate Hidden State $tilde(h)_t$: It is obtained by the current input and the result of the hadamard product of the reset gate and the previous hidden state.
  $
    tilde(h)_t = tanh(W_h dot [r_t dot.o h_(t-1), x_t] + b_h)
    $
+ Hidden State $h_t$: The hidden state is updated by the update gate and the candidate hidden state.
  $
    h_t = (1 - z_t) dot.o h_(t-1) + z_t dot.o tilde(h)_t
    $

For a GRU of $[0,T]$ time steps, the whole forward process can be described as:
#definition[
  #linebreak()
  At timestep $t$:
  + Gate update:
    - Update Gate: $z_t = sigma(W_z dot [h_(t-1), x_t] + b_z)$

    - Reset Gate: $r_t = sigma(W_r dot [h_(t-1), x_t] + b_r)$

  + Candidate Hidden State: $tilde(h)_t = tanh(W_h dot [r_t dot.o h_(t-1), x_t] + b_h)$
  
  + Hidden State: $h_t = (1 - z_t) dot.o h_(t-1) + z_t dot.o tilde(h)_t$

  + (Optional) Output: $y_t = f(W_y h_t + b_y)$
]
