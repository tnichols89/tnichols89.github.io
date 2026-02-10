---
title: 'Deriving Cross Entropy Gradient'
date: 2023-10-18 11:00:00 -0700
categories: [ML, Derivation]
tags: [loss function, classification, supervised training]
math: true
---
## Overview
Cross entropy is a very common loss function used for classification tasks.

Conceptually, cross entropy trains the model to produce a probability distribution over a set of 2 or more classes. Given a sample, the model learns to produce a probability for each possible class from which we can trivially infer the predicted class for the given sample.

Since cross entropy uses softmax to generate a probability distribution over all possible classes for a sample, it is only used for single-classification problems in which every sample must be predicted to have exactly 1 label. It cannot be used for multiple-classification problems where a given sample may have 0 or more than 1 classification. For those types of problems, we use sigmoid and binary cross entropy independently on each possible class for a given sample. That derivation is out of scope of this post.

## Cross Entropy Gradients
Given the following equivalence:

$$
\nabla_{z^{(i)}} \text{CE}(y^{(i)}, \hat{y}^{(i)}) = \nabla_{z^{(i)}} \left( - \Sigma^k_{k=1} \left( y^{(i)}_k \log(\text{softmax}(z^{(i)}_k)) \right) \right)
$$

We want to show:

$$
\begin{align*}
  \nabla_{z^{(i)}} \text{CE}(y^{(i)}, \hat{y}^{(i)}) &= \text{softmax}(z^{(i)}) - y^{(i)} \\
  &= \hat{y}^{(i)} - y^{(i)}
\end{align*}
$$

Where:
- $k$ is the number of possible classes
- $z^{(i)} \in \mathbb{R}^k$ is a $k$-dimensional vector containing unnormalized logits produced by our model for each of $k$ classes
- $z^{(i)}_k \in \mathbb{R}$ is the unnormalized logit produced by our model for the $k^{\text{th}}$ class
- $\hat{y}^{(i)} \in \mathbb{R}^k$ contains a probability distribution over $k$ classes representing the predicted probabilities produced by the model for each class for sample $z^{(i)}$
- $y^{(i)} \in \mathbb{R}^k$ is a one-hot vector with a single $1$ corresponding to the correct class for sample $z^{(i)}$

We proceed by examining the gradient of cross entropy in two cases.

Let $\ell$ denote the index in $y^{(i)}$ of the true label of $z^{(i)}$.

### Case 1
Consider the gradient of cross entropy for all of the incorrect classifications for sample $z^{(i)}$.

That is, assume $j \neq \ell$. Then:

$$
\begin{align*}
  \frac{\partial \text{CE}(y^{(i)}, \hat{y}^{(i)})}{\partial z^{(i)}_j} &= \nabla_{z^{(i)}_j} \left( - \Sigma^k_{\substack{j=1 \\ j \neq \ell}} \left[ y^{(i)}_j \log(\text{softmax}(z^{(i)}_j)) \right] \right) \\
  &= - \Sigma^k_{\substack{j=1 \\ j \neq \ell}} \left[ \frac{y^{(i)}_j}{\text{softmax}(z^{(i)})} \cdot \text{softmax'}(z^{(i)}) \right] \\
  &= 0
\end{align*}
$$

Recall that $y^{(i)}$ is a one-hot vector containing a single $1$ in the location corresponding to the correct class and zeroes elsewhere. Since we are examining the case in which $j \neq \ell$, this sum is computed exclusively over the terms $z^{(i)}$ for which the corresponding $y^{(i)}_j$ terms are zero. 

### Case 2
Consider the gradient of cross entropy for the correct classification for sample $z^{(i)}$.

That is, assume $j = \ell$. Then:

$$
\begin{align*}
  \frac{\partial \text{CE}(y^{(i)}, \hat{y}^{(i)})}{\partial z^{(i)}_j} &= \nabla_{z^{(i)}_j} \left( -y^{(i)}_j \log(\text{softmax}(z^{(i)})) \right) \\
  &= \nabla_{z^{(i)}_j} \left( - \log(\text{softmax}(z^{(i)})) \right) \\
  &= \nabla_{z^{(i)}_j} \left( - \log \left( \frac{e^{z^{(i)}_j}}{\Sigma^k_{n=1} e^{z^{(i)}_n}} \right) \right) \\
  &= \nabla_{z^{(i)}_j} \left( - \left[ \log \left( e^{z^{(i)}_j} \right) - \log \left( \Sigma^k_{n=1} e^{z^{(i)}_n} \right) \right] \right) \\
  &= \nabla_{z^{(i)}_j} \left( - z^{(i)}_j + \log \left( \Sigma^k_{n=1} e^{z^{(i)}_n} \right) \right) \\
  &= -1 + \frac{1}{\Sigma^k_{n=1} e^{z^{(i)}_n}} \cdot \nabla_{z^{(i)}_j} \left[ \Sigma^k_{n=1} e^{z^{(i)}_n} \right] \\
  &= -1 + \frac{1}{\Sigma^k_{n=1} e^{z^{(i)}_n}} \cdot e^{z^{(i)}_j} \\
  &= -1 + \frac{e^{z^{(i)}_j}}{\Sigma^k_{n=1} e^{z^{(i)}_n}} \\
  &= -1 + \text{softmax}_{j}(z^{(i)}) \\
  &= -y_{l}^{(i)} + \hat{y}_{j}^{(i)} \\
  &= \hat{y}_{j}^{(i)} - y_{l}^{(i)}
\end{align*}
$$

### Recombination
Combining the results of our two cases, we see that:

$$
\begin{align*}
  \frac{\partial \text{CE}(y^{(i)}, \hat{y}^{(i)})}{\partial z^{(i)}_j} &= \hat{y}^{(i)} - y^{(i)} + 0 \\
  &= \hat{y}^{(i)} - y^{(i)}
\end{align*}
$$

## Conclusion
We have shown that:

$$
\nabla_{z^{(i)}} \text{CE}(y^{(i)}, \hat{y}^{(i)}) = \hat{y}^{(i)} - y^{(i)}
$$

Despite being an extremely powerful loss function used ubiquitously throughout machine learning, the analytic gradient for cross entropy is actually shockingly simple and efficient to compute.
