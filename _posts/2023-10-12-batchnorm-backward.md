---
title: 'BatchNorm - Part 2: Backward Pass'
date: 2023-10-13 18:30:00 -0700
categories: [ML, Derivation]
tags: [batchnorm, layernorm, backpropagation]
math: true
---
## Goals
In part 2 of our BatchNorm exploration, we will derive the gradients required during the backward pass of the backpropagation method of updating neural network parameters. We assume the reader is familiar with backpropagation concepts.

In order to find the analytic gradients required to carry out the backward pass over BatchNorm during backpropagation, we need to find three partial derivatives: $\frac{\partial f}{\partial \gamma}$, $\frac{\partial f}{\partial \beta}$, and $\frac{\partial f}{\partial x_i}$.

For supplementary reading on this topic, I strongly recommend reading through [Kevin Zakka's blog article](https://kevinzakka.github.io/2016/09/14/batch_normalization/). We both work toward the same end but my approach is organized a bit differently from his.

## Preliminaries
### BatchNorm Recap
Recall that BatchNorm is defined as follows:

$$
\begin{align*}
  \mu_B      &= \frac{1}{m} \Sigma^m_{i=1} x_i \\
  \sigma^2_B &= \frac{1}{m} \Sigma^m_{i=1}(x_i - \mu_B)^2 \\
  \hat{x}_i  &= \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}} \\
  y_i        &= \gamma \hat{x}_i + \beta
\end{align*}
$$

Where:
- $\mu_B, \sigma^2_B, \gamma, \beta, x \in \mathbb{R}^d$
- $d$ represents the number of features in each sample $x$
- $m$ is the number of samples in each minibatch
- $\epsilon \in \mathbb{R}$ is some small number (often `1e-8`) included to avoid divide by zero issues

I recommend going over the `BatchNorm - Part 1: Forward Pass` post if you are in need of a refresher.

### Upstream Gradients
Neural networks are often conceptualized as directed graphs (trees, specifically) in which nodes represent differentiable mathematical operations and edges represent input to or output from a mathematical operation. During the forward pass, data is propagated from the leaves of the tree to the root node, which is a loss function. During the backward pass, gradients are computed starting with the root node (i.e. the loss function) first.

For each node, we compute gradients with respect to each learnable parameter and use those gradients to update the learnable parameters (weights and possibly biases) in each iteration of backpropagation. We also compute gradients with respect to the input to each node even though gradients with respect to the input are not used to update the input to our node directly. Rather, these gradients are propagated to each of the children nodes for use as their respective "upstream gradient" as we apply the chain rule from Calculus to each layer of our neural network.

To this end, we introduce a function $f$ that is not defined in the forward pass for BatchNorm. Since we are computing BatchNorm in the context of a deep neural network and BatchNorm itself is not a loss function, we assume the output from BatchNorm is being passed to a subsequent layer during the forward pass.

We denote this subsequent layer as $f$, and $\frac{\partial f}{\partial y_i}$ represents the upstream gradient.

## Math, Step-by-Step
### Step 1: Find dgamma

$$
\begin{align*}
  \frac{\partial f}{\partial \gamma} &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \gamma} \\
  &= \Sigma^m_{i=1} \frac{\partial f}{\partial y_i} \cdot \hat{x}_i
\end{align*}
$$

Note that we sum over the elements in the mini-batch $m$ because $\gamma$ and $\beta$ are both computed over the each mini-batch.

### Step 2: Find dbeta

$$
\begin{align*}
  \frac{\partial f}{\partial \beta} &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \beta} \\
  &= \Sigma^m_{i=1} \frac{\partial f}{\partial y_i} \cdot \vec{1}
\end{align*}
$$

As we do above, we sum over the elements in the mini-batch $m$ because $\gamma$ and $\beta$ are both computed over the each mini-batch.

### Step 3: Find dx
This part of the problem is not terribly difficult but it is a *lot* of steps. The hardest part here is going to be organizing our work and avoiding repeat work.

We begin with a naive application of the chain rule from multivariate calculus, and we then factor terms out to minimize the number of repeat computations:

$$
\begin{align*}
  \frac{\partial f}{\partial x_i} &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} \\

  &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \left[ \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial \hat{x}_i}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} + \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial x_i} + \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} \right] \\

  &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \left[ \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial x_i} + \left[ \frac{\partial \hat{x}_i}{\partial \mu_B} + \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial \mu_B} \right] \frac{\partial \mu_B}{\partial x_i} \right]
\end{align*}
$$

Note that $\frac{\partial f}{\partial y_i}$ is the upstream gradient, which we assume is given to us. This corresponds to the $\frac{\partial f}{\partial x_i}$ produced by the parent node in the directed computational graph of the neural network during the preceding backpropagation step.

#### Piecemeal Derivations
In an effort to keep our work organized, we break the equation derived above into small pieces and approach them one-by-one before combining them all into the final $\frac{\partial f}{\partial x_i}$.

First, we will need:

$$
\frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i}
$$

But since $\frac{\partial f}{\partial y_i}$ is given to us as the upstream gradient, we will only need to find an expression for $\frac{\partial y_i}{\partial \hat{x}_i}$:

$$
\begin{align*}
  \frac{\partial y_i}{\partial \hat{x}_i} &= \frac{\partial}{\partial \hat{x}_i} \left[ \gamma \hat{x}_i + \beta \right] \\
  &= \gamma
\end{align*}
$$

Second:

$$
\begin{align*}
  \frac{\partial \hat{x}_i}{\partial x_i} &= \frac{\partial}{\partial x_i} \left[ \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}} \right] \\
  &= \frac{\partial}{\partial x_i} \left[ \frac{x_i}{\sqrt{\sigma^2_B + \epsilon}} - \frac{\mu_B}{\sqrt{\sigma^2_B + \epsilon}} \right] \\
  &= \frac{1}{\sqrt{\sigma^2_B + \epsilon}} \\
\end{align*}
$$

Third:

$$
\begin{align*}
  \frac{\partial \hat{x}_i}{\partial \sigma^2_B} &= \frac{\partial}{\partial \sigma^2_B} \left[ \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}} \right] \\
  &= \frac{\partial}{\partial \sigma^2_B} \left[ (x_i - \mu_B)(\sigma^2_B + \epsilon)^{-1/2} \right] \\
  &= -\frac{1}{2} (x_i - \mu_B)(\sigma^2_B + \epsilon)^{-3/2}
\end{align*}
$$

Note that this partial is actually used in two terms. We only want to compute it once and reuse the results for each of the two separate terms.

Fourth:

$$
\begin{align*}
  \frac{\partial \sigma^2_B}{\partial x_i} &= \frac{\partial}{\partial x_i} \left[ \frac{1}{m} \Sigma^m_{i=1}(x_i - \mu_B)^2 \right] \\
  &= \frac{2}{m} (x_i - \mu_B)
\end{align*}
$$

Fifth:

$$
\begin{align*}
  \frac{\partial \hat{x}_i}{\partial \mu_B} &= \frac{\partial}{\partial \mu_B} \left[ \frac{x_i}{\sqrt{\sigma^2_B + \epsilon}} - \frac{\mu_B}{\sqrt{\sigma^2_B + \epsilon}} \right] \\
  &= \frac{-1}{\sqrt{\sigma^2_B + \epsilon}}
\end{align*}
$$

Sixth:

$$
\begin{align*}
  \frac{\partial \sigma^2_B}{\partial \mu_B} &= \frac{\partial}{\partial \mu_B} \left[ \frac{1}{m} \Sigma^m_{i=1} (x_i - \mu_B)^2 \right] \\
  &= \frac{2}{m} \Sigma^m_{i=1} (x_i - \mu_B) (-1) \\
  &= -\frac{2}{m} \Sigma^m_{i=1} (x_i - \mu_B) \\
  &= -2 \left[ \frac{1}{m} \Sigma^m_{i=1} x_i - \frac{1}{m} \Sigma^m_{i=1} \mu_B \right] \\
  &= -2 \left[ \mu_B - \frac{m \cdot \mu_B}{m} \right] \\
  &= -2 \left[ \mu_B - \mu_B \right] \\
  &= 0
\end{align*}
$$

Intuitively, this checks out: we do not, in general, expect the rate of change of the variance to have any dependency on the batch mean. This is also particularly convenient since it zeroes out a chunky term for us.

And, finally:

$$
\begin{align*}
  \frac{\partial \mu_B}{\partial x_i} &= \frac{\partial}{\partial x_i} \left[ \frac{1}{m} \Sigma^m_{i=1} x_i \right] \\
  &= \frac{1}{m}
\end{align*}
$$


#### Tie It All Together
Now that we have the partial derivatives of all the bits and pieces we need, let's apply them to the equation we constructed above using the chain rule:

$$
\begin{align*}
  \frac{\partial f}{\partial x_i} &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} \\

  &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial x_i} + 0 \\

  &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} + \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial y_j} \frac{\partial y_j}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial \mu_B} \right] \frac{\partial \mu_B}{\partial x_i} + \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial y_j} \frac{\partial y_j}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial \sigma^2_B} \right] \frac{\partial \sigma^2_B}{\partial x_i}
\end{align*}
$$

We add the sums over the mini-batches for the $\mu_B$ and $\sigma^2_B$ partials because those are both functions of the *entire* batch of samples, not just the sample $x_i$ for which we are computing the current gradient.

For brevity, let:

$$
\frac{\partial f}{\partial \hat{x}_i} = \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} = \frac{\partial f}{\partial y_i} \gamma
$$

Then:

$$
\begin{align*}
  \frac{\partial f}{\partial x_i} &= \frac{\partial f}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} + \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial \mu_B} \right] \frac{\partial \mu_B}{\partial x_i} + \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \frac{\partial \hat{x}_j}{\partial \sigma^2_B} \right] \frac{\partial \sigma^2_B}{\partial x_i} \\

  &= \frac{\partial f}{\partial \hat{x}_i} \frac{1}{\sqrt{\sigma^2_B + \epsilon}} \\
  &+ \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \frac{-1}{\sqrt{\sigma^2_B + \epsilon}} \right] \frac{1}{m} \\
  &+ \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \left( -\frac{1}{2} \right) (x_i - \mu_B)(\sigma^2_B + \epsilon)^{-3/2} \right] \frac{2}{m} (x_i - \mu_B) \\

  &= \frac{\partial f}{\partial \hat{x}_i} (\sigma^2_B + \epsilon)^{-1/2} \\
  &- \frac{1}{m} (\sigma^2_B + \epsilon)^{-1/2} \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \right] \\
  &- \frac{1}{2} \frac{2}{m} (x_i - \mu_B) \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} (x_j - \mu_B) (\sigma^2_B + \epsilon)^{-1/2} \frac{1}{\sqrt{\sigma^2_B + \epsilon}} \frac{1}{\sqrt{\sigma^2_B + \epsilon}} \right] \\

  &= \frac{\partial f}{\partial \hat{x}_i} (\sigma^2_B + \epsilon)^{-1/2} \\
  &- \frac{1}{m} (\sigma^2_B + \epsilon)^{-1/2} \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \right] \\
  &- \frac{1}{m} (\sigma^2_B + \epsilon)^{-1/2} \frac{(x_i - \mu_B)}{\sqrt{\sigma^2_B + \epsilon}} \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \frac{(x_j - \mu_B)}{\sqrt{\sigma^2_B + \epsilon}} \right] \\

  &= \frac{\partial f}{\partial \hat{x}_i} (\sigma^2_B + \epsilon)^{-1/2} \\
  &- \frac{1}{m} (\sigma^2_B + \epsilon)^{-1/2} \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \right] \\
  &- \frac{1}{m} (\sigma^2_B + \epsilon)^{-1/2} \hat{x}_i \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \hat{x}_j \right] \\

  &= \frac{(\sigma^2_B + \epsilon)^{-1/2}}{m} \left[ m \frac{\partial f}{\partial \hat{x}_i} - \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \right] - \hat{x}_i \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \hat{x}_j \right] \right]
  
\end{align*}
$$

> Note: wherever vector-vector multiplication occurs in these equations, we are calculating the Hadamard product (i.e. elementwise multiplication), not inner or outer products.
{: .prompt-info}

For example,

$$
\frac{\partial f}{\partial \hat{x}_j} \hat{x}_j
$$

denotes elementwise multiplication.

### Bottom Line
Altogether, the three partial derivatives we need to perform the backward pass over BatchNorm are:

$$
\begin{align*}
  \frac{\partial f}{\partial \gamma} &= \Sigma^m_{i=1} \frac{\partial f}{\partial y_i} \cdot \hat{x}_i \\
  \frac{\partial f}{\partial \beta} &= \Sigma^m_{i=1} \frac{\partial f}{\partial y_i} \cdot \vec{1} \\
  \frac{\partial f}{\partial x_i} &= \frac{(\sigma^2_B + \epsilon)^{-1/2}}{m} \left[ m \frac{\partial f}{\partial \hat{x}_i} - \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \right] - \hat{x}_i \Sigma^m_{j=1} \left[ \frac{\partial f}{\partial \hat{x}_j} \hat{x}_j \right] \right]
\end{align*}
$$

Where $\frac{\partial f}{\partial \gamma}$ and $\frac{\partial f}{\partial \beta}$ are used to update the learnable parameters of BatchNorm itself and $\frac{\partial f}{\partial x_i}$ becomes the upstream gradient for the next layer of our network that will be updated during the backward pass.