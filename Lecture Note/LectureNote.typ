#import "@preview/codly:1.2.0": *
#import "@preview/codly-languages:0.1.1": *
#import "@preview/cetz-plot:0.1.1": plot
#show: codly-init.with()
#codly(
  languages:(python:(name:"",icon:"🐍")),
  inset: 4pt,
)
#set heading(
  numbering:("1.")
)
#set page(
  header: [
    Machine Learning
    #line(length:100%)
  ],
  margin: 3cm,
  
)
#let proof(a) = {
  block(
    a,
    inset:8pt,
    radius:8%,
    fill: luma(59.51%, 73.4%)
  )
}

#let algo(title,a) = {
  block(
    text(font:"arial")[
    #line(length:100%)
    #text(10pt)[#title]
    #line(length:100%)
    #a
    #line(length:100%)
    ],
    inset:8pt,
    radius:8%,
    outset:8pt,
  )
}

#align(center)[#text(18pt)[Machine Learning]]
#linebreak()
#import "@preview/cetz:0.3.2": canvas,draw
#set math.mat(delim:"[")
#let example(c) = {
  block(
    c,
    inset:8pt,
    radius:8%,
    fill: luma(230),
  )
}
= Introduction
*Definition*: If a machine's performance,denoted by P, on the task T improves with experience E, then the machine is said to learn from E with respect to T and P.

*Taxonomy* of ML:#align(center)[
  #table(
  columns:2,
  [data type],[learning type],
  [
    + Supervised Learning
    + Unsupervised Learning
    + Reinforcement Learning
    + Semi-supervised Learning
  ],
  [
    + Offline Learning
    + Online Learning
  ]
)
]

*Supervised Learning*: It could help us with two tasks in general. The first if regression and the second is classification. In regression, the output is a continuous value, while in classification, the output is a discrete value.

*Unsupervised Learning*: It is used to find the hidden patterns in the data. It is used in clustering and association.

*Reinforcement Learning*: The agent learns from the feedback it receives from the environment after each action it takes.
#let given = "|"
= Maximum Likelihood Estimation
For an example of multiple cases from the same probability distribution, parameterized by $theta^(hat)$, there exists a $theta$ that maximized the likelihood function of the joint distribution of all the cases, written:
$ limits(theta)^(hat)_"ML" &= limits("argmax")_(theta in (0,1)) space p_(X_1,dots,X_m)(X_1,X_2,dots,X_m;theta) \ &=  limits("argmax")_(theta in (0,1)) space log p_(X_1,dots,X_m)(X_1,X_2,dots,X_m;theta) $ 

Property:
- Consistent: As the number of samples increases, the estimated parameter converges to the true parameter.
- Asymptotically Normal: The estimated parameter is normally distributed as the number of samples increases, and the variance of the distribution decreases as the number of samples increases.

= Linear Regression(With offset)
We define: 
  - m: number of training examples
  - d: number of features
  - $bold(X) in RR^(m times (d+1))$: input matrix, put all the data vector into a matrix
  - $bold(y) in RR^m$: target vector, each entry corresponds to the output of one input vector
  - $bold(w) in RR^(d+1)$: weight vector
  - $bold(b) in RR$: offset number

== Task
Train the model(defined by the parameter, here is a vector) to predict the output vector $bold(y)$ given a new input vector $bold(x)$.

== Procedure
Here we just use the offset version
  + We simply want to find the $bold(w)$ that satisfied: $bold(y) = bold(X)bold(w)$
    In this step, we may decide whether the linear system has a solution or not. If not, we may use the least square solution.
  + Define notation $f_(bold(w),b)(bold(x)_i) = bold(w)^T bold(x) + b$, and define the loss for each example $ e_i =f_(bold(w),b)(bold(x)_i)-y_i $
    Then the loss function is $L(bold(w),b) = 1/m sum_(i=1)^m e_i^2 = 1/m sum_(i=1)^m (f_(bold(w),b)(bold(x)_i)-y_i)^2$
  + The least square solution that minimize the loss function is $bold(w) = (bold(X)^T bold(X))^(-1) bold(X)^T bold(y)$, (without offset, just use the original input matrix $bold(X)$). Be careful whether the matrix $bold(X)^T bold(X)$ is invertible or not, i.e., $bold(X)$ is full column rank.
  + Then we could predict new input $bold(y_"new") = bold(x_"new")^T bold(w)$.
= Linear Regression with Multiple Outputs
We define: 
  - m: number of training examples
  - d: number of features
  - h: number of outputs features
  - $bold(X) in RR^(m times (d+1))$: input matrix, $bold(X) = mat(bold(1), bold(X'))$
  - $bold(Y) in RR^(m times h)$: output matrix
  - $bold(W) in RR^((d+1) times h)$: design matrix, $bold(W) = mat(bold(b)^T;bold(W'))$
  - $bold(b) in RR^h$: offset vector, we ignore it and put it into the design matrix

== Task
Train the model(defined by the parameter, here is a matrix) to predict the output vector $bold(y)$ given a new input vector $bold(x)$.

== Procedure
  The procedure is similar to the single output case. For the first step, we could devide the output matrix into h single output vectors, try to solve the linear system for each output vector. If the matrix $bold(X)$ is full column rank, we may get the least square solution.
  + Define loss function: 
    $ "Loss"(bold(W)) = "Loss"(bold(W),bold(b))= sum_(k=1)^h (bold(X) bold(w)_k - bold(y)^(k))^T (bold(X) bold(w)_k - bold(y)^(k))$
  + The least square solution is $bold(W) = (bold(X)^T bold(X))^(-1) bold(X)^T bold(Y)$, where $bold(X) = mat(bold(1), bold(X'))$.

== MLE and Linear Regression

We have $y_i = bold(w)^T bold(x)_i + e_i$ ($b$ is in the vector $bold(w)$). The error term $e_i$ is of gaussian distribution and therefore $y_i|bold(x)_i;bold(w),sigma^2 ~ N(bold(w)^T bold(x)_i, sigma^2)$, where $sigma^2$ is the variance of the error term. We could write the likelihood function as:
$ L(bold(w),sigma^2 | {y_i,bold(x)_i}) = product_(i=1)^n 1/(sqrt(2 pi sigma^2)) exp (-(y_i - bold(w)^T bold(x)_i)^2 / (2 sigma^2)) $

Solve the MLE problem
+ $log(L(bold(w),sigma^2) given {y_i,bold(x_i)}) = -n/2log(2pi sigma^2) - 1/(2 sigma^2)sum(y_i-bold(w)^T bold(x)_i)^2$
+ use the derivative to find the maximum likelihood estimator: $ sigma/(sigma bold(w)) log(L(bold(w),sigma^2) given {y_i,bold(x_i)}) =1/(sigma^2) sum_(i=1)^n (y_i-bold(w)^T bold(x_i))bold(x)_i =0 $ 
+ we could get the least square solution: $bold(w) = (bold(X)^T bold(X))^(-1) bold(X)^T bold(y)$. In fact, the least square solution is the MLE solution. (The decution process is ignored here)

#let sign = "sign"
= Linear Classification
Main idea: To treat binary classification as linear regression in which the output $y_i$ is binary. $y_i in {-1,+1}$.
- Learning and training part: Similar procedure, obtain the weight vector $bold(w)$.
- Prediction part: $y_"new" = sign(bold(x)_("new")^T  bold(w)) = sign(mat(1;bold(x')_("new"))^T bold(w)) in {-1,+1} $

The $sign$ function is defined as:
$ sign(x) = cases(+1 "if" x>=0, -1 "otherwise") $

== Python Demo
```python
import numpy as np
from numpy.linalg import inv  

X = np.array([[1,-7],[1,-5],[1,1],[1,5]])
y = np.array([[-1],[-1],[1],[1]])
#linear regression for classification
w = inv(X.T @ (X)) @ (X.T) @ (y)
print(w,"\n")

#predict
X_new = np.array([[1,2]])
y_predict_new = np.sign(X_new @ w)
print(y_predict_new)
# expected output: [[-1.]]
```

== Multi-class Classification
Idea: one-hot encoding, for classes ${1,2,dots,C}$, where $C>2$ is the number of classes. The correspoding label vector is
$ &bold(y)_("c1") = mat(1,0,0,dots,0) \ &bold(y)_("c2") = mat(0,1,0,dots,0) \ &dots.v \ &bold(y)_("cC") = mat(0,0,0,dots,1) $

We store the class vectors of datasets into a label matrix $bold(Y) in RR^(m times C)$, where $m$ is the number of training examples. It is a binary matrix. Essentially we are doing C separate linear classification problems with class $k$ and other classes as a single class.

- Learning and training part: Similar procedure, obtain the weight matrix $bold(W) in RR^((d+1) times C)$.
- Prediction part: $ bold(y)_("new") = limits("argmax")_(k in {1,2,dots,C})(bold(x)_("new")^T bold(W[:,k])) $ in which the $bold(W)[:,k] in RR^(d+1)$ is the $k$-th column of the weight matrix.
=== Python Demo
```python
import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import OneHotEncoder
# manually encode the label matrix
X = np.array([[1,1,1],[1,-1,1],[1,1,3],[1,1,0]])
Y_class = np.array([[1],[2],[1],[3]])
Y = np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1]])

# one-hot encoding function
Y_onehot = OneHotEncoder().fit_transform(Y_class).toarray()

# learning
W = inv(X.T @ X) @ X.T @ Y # could use Y_onehot instead of Y

# predicting
X_new = np.array([[1,0,-1]])
Y_predict = X_new @ W
print(np.argmax(Y_predict)+1)
```
#let new = "new"
= Polynomial Regression
== Motivation
- The data may come from a polynomial function, but the linear model may not be able to fit the data well.
- For classification, the $xor$ dataset is not linearly separable, and polynomial function could separate the data.
== Idea
- We need some knowledge about polynomial functions.
  + Let $f_bold(w)(bold(x)) = w_0 + w_1 x_1 + dots + w_d x_d$, then it has 1 variable and d degree. Let $f_bold(w)(bold(x)) = w_0 + w_1 x + w_2 x_2 + w_3 x_1 x_2+ w_4 x_1^2 +w_5 x_2^2$, then it has 2 variables and 2 degree.
   each term is called monomial.
  + For a polynomial function with d degree and n variables, the number of monomials is $mat(d+n;n;delim:"(")$

*Training*
- For $m$ data, $n$ order and $d$ variables, the polynomial matrix is $ P = mat(bold(p)_1^T;bold(p)_2^T;dots.v;bold(p)_m^T) in RR^(m times mat(d+n;n;delim:"(")) $ and $bold(p)_i = mat(1, m_1,m_2, dots, m_k)^T$, $m_i$ is monomial and $k = mat(d+n;n;delim:"(")$
- weight vector $bold(w) = P^T (P P^T)^(-1) bold(y)$ (Dual form of the least square solution before) 
*Predicting*
- $y_"new" = P_"new" bold(w)$

== Polynomial Classification
- Similar to the linear Classification
- For simple output, $y_"new" = sign(bold(p)_"new"^T bold(w))$
- For multiple outputs, $bold(y)_new = limits("argmax")_(k in {1,2,dots,c}) (bold(p)_new^T bold(W[:,k]))$

= Ridge Regression
#let sol = $bold(X)^T bold(X) + lambda bold(I)$
#let tsp(x) = $bold(#x)^T$
#let vec(x) = $bold(#x)$
== Motivation
In real life, the features $d$ is large and the number of examples $m$ is small, so the matrix *$X^T X in RR^(d times d)$* is hard to invert. So we need to stabilize and robustify the solution.
== Idea
Recall the loss function of linear regression and introduce the regularization term(ridge regression version):
$ L(bold(w),b) &= 1/m sum_(i=1)^m (f_(bold(w),b)(bold(x)_i)-y_i)^2 = (bold(X) bold(w) - bold(y))^T (bold(X) bold(w) - bold(y)) \ J(bold(w),b) &= L(bold(w),b) + lambda ||bold(w)||_2^2 \ &= 1/m sum_(i=1)^m (f_(bold(w),b)(bold(x)_i)-y_i)^2 + lambda ||bold(w)||_2^2 \ &=  (bold(X) bold(w) - bold(y))^T (bold(X) bold(w) - bold(y)) + lambda bold(w)^T bold(w) $
where $lambda$ is the regularization parameter. We need to minimize the loss function, that is to find:
$ bold(w)^* = limits("argmin")_(bold(w)) ((bold(X) bold(w) - bold(y))^T (bold(X) bold(w) - bold(y)) + lambda bold(w)^T bold(w))  $ and the result is $ bold(w)^* = (bold(X)^T bold(X) + lambda bold(I)_(d+1))^(-1) bold(X)^T bold(y) $
The term $bold(X)^T bold(X) + lambda bold(I)$ is always invertible because it is positive definite. The predicting part is the same as the linear regression.

#proof([
  To prove $sol$ is invertible, we need to prove it is positive definite. By definition of positive definite, we need to prove: $forall bold(x), bold(x)^T bold(A) bold(x) <-> bold(A) "is positive definite."$ We have
  $ bold(x)^T (sol) bold(x) &= bold(x)^T bold(X)^T bold(X) bold(x) + tsp(x) lambda bold(I) bold(x) \ &= tsp((bold(X) bold(x))) (bold(X) bold(x)) + tsp(vec(x)) vec(x)  $

  Since every term is positive, then the result is larger or equal to zero. $sol$ is positive definite and therefore always invertible.
])

However, this term is in $RR^((d+1) times (d+1))$ so it is hard to find the inverse of it. So we could use the dual form of the solution which needs to find the inverse of a matrix in $RR^(m times m)$.
$ bold(w) = bold(X)^T (bold(X) bold(X)^T + lambda bold(I)_m)^(-1)  bold(y) $

The proof of the dual form need to use Woodbury formula:

$ (vec(I)+vec(U)vec(V))^(-1) = vec(I) - vec(U)(vec(I)+ vec(U)vec(V))^(-1) vec(V) $

== Python Demo
Refer to #link("https://github.com/c-allergic/DSAA-2011-Demo",text(blue)[DSAA-2011-Demo])

= Gradient Descent
== Motivation
In linear regression and the other few models mentioned above, the optimal solution could be simply found by solving the equation. However, in many cases, to minimize the loss function with respect to parameter $w$ is hard. So we want an algorithm that could iteratively find the optimal solution, that is the gradient descent algorithm.

== Idea
#let Der(a,b,c:$sigma$) = $(#c #a)/(#c #b)$
- Task: Minimize the loss function $C(vec(w))$ in which $vec(w) = mat(w_1,dots,w_d)^T$
- Gradient of $vec(w)$: 
  $gradient_vec(w) C(vec(w)) =  mat(Der(C,w_1),Der(C,w_2),dots,Der(C,w_d))^T$.
  It's a function or say a vector of $vec(w)$. And the direction of the gradient is the direction of the fastest *increase* of the function, while the opposite direction is the direction of the fastest *decrease* of the function.

- Algorithm
  + Initial $vec(w)$ with learning rate $eta > 0$
  + $vec(w)_(k+1) = vec(w)_k - eta gradient_vec(w) C(vec(w)_k)$
  + Repeat the above step until the convergence condition is satisfied.

- Convergence Criterias:
  + The absolute or percentage change of the loss function is smaller than a threshold.
  + The absolute or percentage change of parameter $vec(w)$ is smaller than a threshold.
  + Set a maximum number of iterations.
- Notice that according to multivariate calculus, if $eta$ is not too large,$C(vec(w)_(k+1)) < C(vec(w)_k)$. But GD could only find the local minimum.

== Variation of GD
#let gw(x) = $gradient_vec(w) C(vec(w)_#x)$
=== Change the learning rate
+ Decreasing learning rate $ eta = eta_0 / (1 + k) $ where $k$ is the number of iterations or other form of changes.
  It could help the algorithm to converge faster at the beginning and avoid oscillation at the end.
+ Adaptive learning rate
  $ vec(w)_(k+1) = vec(w_k) - eta/(sqrt(G_k + epsilon)) gradient_vec(w) C(vec(w)_k) $
  where $G_k = sum_(i=0)^k norm(gw(i))_2^2$ and $epsilon$ is a small positive number to avoid zero denominator. This method gives larger update to smaller gradient and vice versa
  , adjusting the learning rate according to the history information. But the learning rate may shirnk too fast.
=== Different gradient
+ Momentum-based GD $ &vec(v)_k = beta vec(v)_(k-1) + (1-beta)gw(k-1) \ &vec(w)_(k+1) =  vec(w)_k - eta vec(v)_k $
  where $beta in (0,1)$ is the momentum parameter and $vec(v)_0 = gw(0)$. It converges fast but may overshoot the optimal point.
+ Nesterov Accelerated Gradient (NAG)
  $ &vec(v)_k = beta vec(v)_(k-1) + eta gradient_vec(w) C(vec(w)+beta vec(w)_(k-1)) \ &vec(w)_(k+1) =  vec(w)_k - vec(v)_k $
  where $vec(v)_0 = 0$.It works by anticipating the next direction of the optimizer. It is fast but complex.

=== Design of loss function
 
The idea is only calculate the loss of some sample from the dataset, reduce the computation cost.
+ Batch GD: Use all the data to calculate the gradient.
+ Stochastic GD: Use one randomly chosen data to calculate the gradient.
+ Mini-batch GD: Use a small batch of randomly chosen data to calculate the gradient.

= Logistic Regression
== Motivation
There are some possible issues for classification problems: 
+ Noises: Lead to unseparable data.
+ Mediocre generalization: Only find barely boundary.
+ Overfitting: The model is too complex and fits the noise in the data.

In that case, we want a model that output our confidence of the prediction and that's why we need logistic regression.

#let dotp(x,y) = $#sym.quote.angle.l.single #x,#y #sym.quote.angle.r.single$
== Idea
- Logistic function: $ g(z) = 1/(1+e^(-z)) $
- Logistic Regression: In binary classification task, we set our prediction $ Pr(y=1|bold(x),bold(theta),theta_0) = g(dotp(vec(theta),vec(x))+theta_0) = 1/(1+e^(-(dotp(vec(theta),vec(x))+theta_0))) $
  This is known as class conditional probability.

#proof([
  How could we derive the form of the class conditional probability? We first define the log-odds function of both classes as a affine function of the input vector $bold(x)$:
$ log Pr(y=1|vec(x),vec(theta),theta_0)/Pr(y=-1|vec(x),vec(theta),theta_0) = vec(theta) vec(x) + theta_0 $
Set $Pr(y=-1|vec(x),vec(theta),theta_0) = 1- Pr(y=1|vec(x),vec(theta),theta_0)$, then:
$ log Pr(y=1|vec(x),vec(theta),theta_0)/(1- Pr(y=1|vec(x),vec(theta),theta_0)) $

Let $Pr(y=1|vec(x),vec(theta),theta_0) = a, dotp(vec(theta),vec(x)) = b$, then it is equivalent to: 
$  &log a/(1-a) = b  => a/(1-a) = e^b => a = (1-a) e^b => a=e^b/(1+e^b)  => a = 1/(1+e^(-b)) $ 
So $Pr(y=1|vec(x),vec(theta),theta_0) =  1/(1+e^(-dotp(vec(theta),vec(x)))) $

])
- $dotp(vec(theta),vec(x)) + theta_0 = 0$ is the decision boundary. If $dotp(vec(theta),vec(x)) + theta_0 > 0$, then $Pr(y=1|vec(x),vec(theta),theta_0) > 0.5$ and vice versa.

= Support Vector Machine
== Get familiar with Logistic Regression

Logistic regression is a binary classification model.
$ f_vec(w)(vec(x)) =  1/(1+exp(-vec(w)^T vec(x))) $

The loss function in logistic regression is $ L(bold(w)) = -sum_(i=1)^m y_i log(f_(bold(w))(bold(x)_i)) + (1-y_i) log(1-f_(bold(w))(bold(x)_i)) + lambda/(2m) sum_(j=1)^n w_j^2 $




The loss function shows that if the label is 1, then the loss is $-log(f_(bold(w))(bold(x)_i))$, meaning that $f_vec(w)(vec(x)_i)$ or say $- vec(w)^T vec(x)$ should be as large as possible and vice versa. Note that the $lambda$ could be understood as how much we want to penalize the large $vec(w)$, if we care more about the loss of each example, we just set $lambda$ small.

== Idea
We set the term $-log(1/(1+exp(-vec(w)^T vec(x))))$ as $"cost"_1(vec(w)^T vec(x))$ and the term $-log(1-1/(1+exp(-vec(w)^T vec(x))))$ as $"cost"_0(vec(w)^T vec(x))$. The cost function for SVM is then: $ L(vec(w)) = C sum_(i=1)^m "cost"_(y_i)(vec(w)^T vec(x)_i) + 1/2 sum_(j=1)^n w_j^2 $

Constrast to the loss function of logistic regression, we change the cost term and the constant parameter before each term. 

#align(center)[
  #table(
  columns:2,
  [$"cost"_1$ and $-log(1/(1+e^(-x)))$ ],[$"cost"_0$ and $-log(1-1/(1+e^(-x)))$],
  [
    #let f_w(x) = -calc.log(1/(1+calc.pow(calc.e,-x)))
#figure(canvas({
  import plot: *
  plot(
    size:(3,3),
    x-tick-step: 1,
    y-tick-step: .4,
    axis-style: "left",
    {
      add(f_w,domain:(-2,2.5))
      add(((-2,.8),(1,0),(2,0)))
      
    }
  )
}))],[#let f_w(x) = -calc.log(1- (1/(1+calc.pow(calc.e,-x))))
#figure(canvas({
  import plot: *
  plot(
    size:(3,3),
    x-tick-step: 1,
    y-tick-step: .4,
    axis-style: "left",
    {
      add(f_w,domain:(-2.5,2))
      add(((-2,0),(-1,0),(2,.8)))
    }
  )
}))])]

== Nickname: Large Margin Classifier
When classify the data, classifier would figure out a boundary that could separate the data. The margin is defined as the distance between the boundary and the nearest point of the data. The boundary SVM figure out are the one that has the largest margin. 

When the parameter of example cost $C$ is really large, the classifier would be snesitive to the outliers, which may lead to overfitting. The reason why SVM would act like this lies in the optimization problem of the cost function(SVM tend to set the cost really small when $C$ is large).


