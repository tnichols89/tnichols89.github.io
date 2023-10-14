---
title: 'BatchNorm - Part 1: Forward Pass'
date: 2023-10-13 19:00:00 -0800
categories: [ML, Derivation]
tags: [batchnorm, layernorm, backpropagation]
math: true
---
## Goals
In this post, we will be seeking to understand batch normalization, which we will refer to as BatchNorm for brevity. We will specifically be focusing on understanding what BatchNorm does during the forward pass, as well as deriving the equations used during the backward pass in backpropagation.

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
BatchNorm is articulated by Sergey Ioffe and Christian Szegedy in [the paper that originally introduced it](https://proceedings.mlr.press/v37/ioffe15.html) as follows:

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
BatchNorm and LayerNorm are both frequently used to stabilize deep neural network training and mitigate issues with vanishing and exploding gradients.

Implementations of each are nearly identical, and you can implement LayerNorm by transposing the feature matrix prior to passing it through a BatchNorm implementation. The implementation does not need to maintain a running mean and variance for LayerNorm as it does for BatchNorm.

BatchNorm and LayerNorm are functionally identical, with a couple notable exceptions:
- BatchNorm normalizes features in a sample vector on a per-feature basis rather than on a per-sample basis, using mean and variance vectors computed over entire mini-batches rather than mean and variance computed over just the elements of a single sample
  - BatchNorm is thus sensitive to batch size during training whereas LayerNorm is not
  - BatchNorm also does not tend to scale up as well as LayerNorm does
- BatchNorm maintains running mean and variance variables that are updated with each mini-batch to develop mean and variance vectors representative of the overall dataset
  - LayerNorm does not have to maintain running mean and variance as the mean and variance are computed across the features of each sample independently of other samples

In addition to z-score normalization, BatchNorm and LayerNorm also include the benefit of adding two learned parameters, $\gamma$ and $\beta$, which can theoretically be learned to entirely undo the z-score normalization, if that's what the gradients suggest is optimal during backpropagation. These two learnable parameters also simply improve the expressiveness of the overall model.