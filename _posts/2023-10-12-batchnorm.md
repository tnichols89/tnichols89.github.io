---
title: Understanding Batch Normalization
date: 2023-10-11 20:00:00 -0800
categories: [AI, ML]
tags: [backpropagation, derivation]
math: true
---
## Goals
In this post, we will be seeking to understand batch normalization, which we will refer to as BatchNorm for brevity. We will specifically be focusing on understanding what BatchNorm does during the forward pass, as well as deriving the equations used during the backward pass in backpropagation. Code examples will be given in NumPy but are easily adaptable to most other modern numeric processing libraries such as JAX or PyTorch.

We assume that you are familiar with a few things:
- The chain rule in calculus
- Partial derivatives
- Basic linear algebra
  - Vector-matrix and matrix-matrix multiplication
- Backpropagation concepts

We generally assume that if you are reading this article, you are very specifically looking for help with understanding backpropagation through a BatchNorm layer.

## BatchNorm Overview
BatchNorm is a common normalization technique used in deep neural networks to stabilize training and mitigate issues with exploding and vanishing gradients. It is often applied to output from a neural network layer prior to feeding the output forward as input to the next layer.

### BatchNorm vs LayerNorm
BatchNorm is also trivially adapted into "layer normalization," which I'll refer to as LayerNorm for brevity, by simply transposing the input feature matrix and correspondingly transposing the output of the BatchNorm function. BatchNorm is often seen in neural networks designed for computer vision and LayerNorm is often found in transformers.

Where BatchNorm performs a "feature-wise" normalization of the elements/features in a vector using the per-feature mean and variance computed across all samples in a (mini-)batch, LayerNorm normalizes elements in each sample/vector using the mean and variance computed over just the elements of that one sample/vector. That is, when using LayerNorm, each sample/vector is normalized independently of the others, whereas BatchNorm normalizes samples/vectors using batch-wide mean and variance.

BatchNorm performance is thus sensitive to batch sizes whereas LayerNorm is not.

## Forward Pass
### Z-score Normalization Recap
Before we mathematically define BatchNorm, recall that z-score normalization is defined as:

$$
\hat{x} = \frac{x - \mu}{\sigma}
$$

Where:
- $\hat{x}, x, \mu, \sigma \in \mathbb{R}$
- $x$ represents a scalar sample from a population
- $\mu$ represents the scalar mean of the population
- $\sigma$ represents the scalar standard deviation of the population
- $\hat{x}$ represents the z-score normalized value corresponding to $x$

When we apply z-score normalization to a batch of samples, the resulting batch has a mean of zero and a standard deviation of one ("unit standard deviation").

This is the basis upon which BatchNorm was designed.

### BatchNorm Definition
BatchNorm is articulated by Sergey Ioffe and Christian Szegedy in [the paper that originally introduced it](http://proceedings.mlr.press/v37/ioffe15.html) as follows:

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

In plain terms:
- $\mu_B$ is a vector of the same length as $x$ containing the per-feature mean across the entire (mini-)batch
- $\sigma^2_B$ is a vector of the same length as $x$ containing the per-feature variance across the entire (mini-)batch
- $\hat{x}_i$ is the per-feature, batch-normalized feature vector corresponding to the $i$th sample $x_i$
- $y_i$ is the batch-normalized $\hat{x}_i$ complemented with learnable parameters $\gamma$ and $\beta$ corresponding to the $i$th sample $x_i$

### Adding Learnable Parameters
Notice that BatchNorm is simply doing z-score normalization but with vectors plus some extra magic. In addition to the z-score normalization that has been adapted to work on vectors, BatchNorm introduces two *learnable parameters*: $\gamma$ and $\beta$. When we say *learnable parameters*, we are referring to the fact that $\gamma$ and $\beta$ will be iteratively updated with every backpropagation pass.

The introduction of these learnable parameters achieves two things: it adds expressivity to the neural network, and it allows the network to *learn to undo the normalization*.

Imagine the model determined via backpropagation that optimal $\gamma$ and $\beta$ values for the given dataset were:

$$
\begin{align*}
  \gamma &= \sqrt{\sigma^2_B + \epsilon} \\
  \beta  &= \mu_B
\end{align*}
$$

Then:

$$
\begin{align*}
  y_i &= \gamma \left( \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}} \right) + \beta \\
      &= \sqrt{\sigma^2_B + \epsilon} \left( \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}} \right) + \mu_B \\
      &= x_i - \mu_B + \mu_B \\
      &= x_i
\end{align*}
$$

We can see that, by adding $\gamma$ and $\beta$ as learnable parameters, the neural network is able to determine what the optimal degree of normalization is based on gradients computed during backpropagation. It is absolutely conceivable that the network learns that normalization actually does not help on a given dataset and therefore effectively stops normalizing the data.

In most cases, though, it seems safe to assume that the model will benefit from at least some degree of normalization and learn $\gamma$ and $\beta$ accordingly.

### Key Insights
BatchNorm and LayerNorm are functionally identical, with the two exceptions that:
- BatchNorm normalizes features in a sample vector on a per-feature basis rather than on a per-sample basis, using mean and variance vectors computed over entire mini-batches rather than mean and variance computed over just the elements of a single sample
  - BatchNorm is thus sensitive to batch size during training whereas LayerNorm is not
  - BatchNorm also does not tend to scale up as well as LayerNorm does
- BatchNorm maintains running mean and variance variables that are updated with each mini-batch to develop mean and variance vectors representative of the overall dataset

In addition to z-score normalization, BatchNorm and LayerNorm also include the benefit of adding two learned parameters, $\gamma$ and $\beta$, which can theoretically be learned to entirely undo the z-score normalization, if that's what the gradients suggest is optimal during backpropagation. These two learnable parameters also simply improve the expressiveness of the overall model.

## Backward Pass
### Goal
In order to find the analytic gradients required to carry out the backward pass over BatchNorm during backpropagation, we need to find three partial derivatives: $\frac{\partial f}{\partial \gamma}$, $\frac{\partial f}{\partial \beta}$, and $\frac{\partial f}{\partial x_i}$.

### Preliminaries
Let's clarify a few things before moving forward. We have introduced some function $f$ that is not defined in the forward pass for BatchNorm. Since we are computing BatchNorm in the context of a deep neural network, we are making a couple key assumptions:
- During the forward pass, the output from BatchNorm is being passed as input to the next layer
  - This implies that, during the backward pass, there exists an upstream gradient (upstream relative to the BatchNorm node) originating from the parent node of the BatchNorm node corresponding to the next layer to which the output from BatchNorm is passed in the directed computational graph for the neural network that must be included in our calculations
- Even though the only two learnable parameters in BatchNorm are $\gamma$ and $\beta$, we must compute the partial derivate with respect to $x_i$ as well because *that* becomes the upstream gradient originating from the BatchNorm node that flows to the layer that is computed prior to BatchNorm during the forward pass
  - That is, $\frac{\partial f}{\partial x_i}$ becomes the upstream gradient used in the backward pass computations for whatever layer produces the data that is used as input to BatchNorm during the forward pass

### Math, Step-by-Step
#### Step 1: Find dgamma

$$
\begin{align*}
  \frac{\partial f}{\partial \gamma} &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \gamma} \\
  &= \Sigma^m_{i=1} \frac{\partial f}{\partial y_i} \cdot \hat{x}_i
\end{align*}
$$

Note that we sum over the elements in the mini-batch $m$ because $\gamma$ and $\beta$ are both computed over the each mini-batch.

#### Step 2: Find dbeta

$$
\begin{align*}
  \frac{\partial f}{\partial \beta} &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \beta} \\
  &= \Sigma^m_{i=1} \frac{\partial f}{\partial y_i} \cdot \vec{1}
\end{align*}
$$

As we do above, we sum over the elements in the mini-batch $m$ because $\gamma$ and $\beta$ are both computed over the each mini-batch.

#### Step 3: Find dx
This part of the problem is not terribly difficult but it is a *lot* of steps. The hardest part here is going to be organizing our work and avoiding repeat work.

We begin with a naive application of the chain rule from multivariate calculus, and we then factor terms out to minimize the number of repeat computations:

$$
\begin{align*}
  \frac{\partial f}{\partial x_i} &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial x_i} + \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} \\

  &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \left[ \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial \hat{x}_i}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} + \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial x_i} + \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial \mu_B} \frac{\partial \mu_B}{\partial x_i} \right] \\

  &= \frac{\partial f}{\partial y_i} \frac{\partial y_i}{\partial \hat{x}_i} \left[ \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial x_i} + \left[ \frac{\partial \hat{x}_i}{\partial \mu_B} + \frac{\partial \hat{x}_i}{\partial \sigma^2_B} \frac{\partial \sigma^2_B}{\partial \mu_B} \right] \frac{\partial \mu_B}{\partial x_i} \right]
\end{align*}
$$

Note that $\frac{\partial f}{\partial y_i}$ is the upstream gradient, which we assume is given to us. This corresponds to the $\frac{\partial f}{\partial x_i}$ produced by the parent node in the directed computational graph of the neural network during the preceding backpropagation step. In code, this would be represented by a tensor passed in as an argument that was produced in the preceding backpropagation step.

Before we evaluate the partials, we list out each "atomic" piece of the puzzle derived above that we need to compute in order to actually calculate $\frac{\partial f}{\partial x_i}$.

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

#### Step 4: Plug 'n Chug

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