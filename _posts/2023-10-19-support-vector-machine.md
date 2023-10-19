---
title: 'SVMs and the Hinge Loss Function'
date: 2023-10-19 11:30:00 -0700
categories: [ML, Derivation]
tags: [loss function, classification, supervised training]
math: true
---
## Overview
Support vector machines (SVMs) are arguably the most powerful linear classification models in widespread use. They are linear models that can be trained using the hinge loss function, among other similar linear loss functions.

SVMs are able to work so well in part because the hinge loss function - a loss function commonly used by SVMs - incentivizes the model to generate scores (i.e. unnormalized logits produced by the model) with a hyperparameter margin $\Delta$ between the scores corresponding to the various possible classifications that may be applied to each sample. In addition, high scores for correct classification of a given sample are rewarded and, simultaneously, high scores for incorrect classification of the given sample are harshly penalized.

The margin concept may seem a bit abstract at first, but its utility becomes more apparent as we dive into the definition of the hinge loss function used by SVMs.

SVMs can also handle linearly separable data when trained with the hinge loss function.

Despite being linear models, SVMs can utilize nonlinear "kernels" that further improve their effectiveness against datasets with certain nonlinear distributions. Commonly used kernels include the dot product kernel, the polynomial kernel, the radial basis function kernel, and the sigmoid kernel.

The hinge loss function can also be articulated with arbitrary custom kernels. Optimal kernels must meet certain linear algebraic criteria that is out of scope of this article.

In this post, we will be diving into the vanilla formulation of the hinge loss function based on the dot product. This formulation is broadly applicable to many problem types. 

## Hinge Loss Function
### Definition
Similar to a logistic regression model, the hinge loss function has a separate weight vector for each possible classification.

The hinge loss function can be formalized as:

$$
\begin{align*}
  \ell(x_i) &= \Sigma_{\substack{j \in K \\ j \neq y_i}} \max(0, w^T_j x_i - w^T_{y_i} x_i + \Delta) \\
  &= \Sigma_{\substack{j \in K \\ j \neq y_i}} \begin{cases}
    w^T_j x_i - w^T_{y_i} x_i + \Delta \quad & \text{if } w^T_j x_i - w^T_{y_i} x_i + \Delta > 0 \\
    0 \quad & \text{otherwise}
  \end{cases}
\end{align*}
$$

Where:
- $K$ is the set of all possible classifications that may be applied to any given sample
- $x_i \in \mathbb{R}^d$ is a given sample
- $y_i$ denotes the ground truth classification for the given sample $x_i$
- $w_{y_i} \in \mathbb{R}^d$ is the weight vector corresponding to the correct classification of the given sample $x_i$
- $w_j \in \mathbb{R}^d$ is the weight vector for each incorrect classification with respect to the given sample $x_i$
- $\Delta \in \mathbb{R}$ is an arbitrary hyperparameter constant, commonly chosen as $\Delta = 1$

> This loss function encourages $w^T_{y_i} x_i$ to be as high-magnitude positive as possible and $w^T_j x_i$ to be as high-magnitude negative as possible. The loss goes to zero once $w^T_{y_i} x_i > w^T_j x_i$ by at least the margin $\Delta$ for all negative classifications $j$ of sample $x_i$, indicating the objective has been optimally achieved.
{: .prompt-tip}

### Gradients
The hinge loss function contains two learnable parameters - $w_j$ and $w_{y_i}$ - so we have to derive gradients for both.

Although the hinge loss function is not, strictly speaking, differential due to the incontinuity introduced by the $\max$ function at $w^T_j x_i - w^T_{y_i} x_i + \Delta = 0$, this does not cause a problem in practice. The function is rarely ever evaluated precisely at 0 due to floating point imprecision and so the gradients are unlikely to become undefined, but implementations also typically set the gradients at $0$ to $0$ manually even though they are technically undefined.

#### Case 1
We begin by deriving the gradient with respect to the weight vector corresponding to the ground truth classification $w^T_{y_i}$ for any given sample $x_i$:

$$
\begin{align*}
  \frac{\partial \ell(x_i)}{\partial w_{y_i}} &= \Sigma_{\substack{j \in K \\ j \neq y_i}} \begin{cases}
    -x_i \quad & \text{if } w^T_j x_i - w^T_{y_i} x_i + \Delta > 0 \\
    0 \quad & \text{otherwise}
  \end{cases} \\
  &= - \left( \Sigma_{\substack{j \in K \\ j \neq y_i}} \left[ \mathbb{1} \left\{ w^T_j x_i - w^T_{y_i} x_i + \Delta > 0 \right\} \right] \right) x_i
\end{align*}
$$

In case you have not encountered it before:

$$
\mathbb{1} \left\{ \text{<conditional>} \right\}
$$

is referred to as the "indicator function". It simply evaluates to the number to the left ($\mathbb{1} in this case$) when the `<conditional>` inside of the curly braces evaluates to true. This notation is often used for counting in ML texts.

> This gradient counts the number of incorrect classifications with scores that were not at least $\Delta$ lower than the correct classification score and scales the negation of the sample feature vector $x_i$ by that value. The optimizer - typically stochastic gradient descent with a linear model like this - will then use that value to either update $w^T_{y_i}$ to yield a higher score or not make any changes at all if $w^T_{y_i} x_i > w^T_j x_i$ by at least $\Delta$ for all negative classifications $j$.
{: .prompt-info}

#### Case 2
We now derive the gradient with respect to the weight vectors corresponding to the incorrect classifications $w^T_j$ for any given sample $x_i$:

$$
\frac{\partial \ell(x_i)}{\partial w_{j}} = \begin{cases}
  x_i \quad & \text{if } w^T_j x_i - w^T_{y_i} x_i + \Delta > 0 \\
  0 \quad & \text{otherwise}
\end{cases}
$$

> For a given sample $x_i$ and a single weight vector $w^T_j$ corresponding to one of the incorrect classifications of $x_i$, the optimizer will use this gradient to either update $w^T_j$ in such a way that $w^T_j x_i$ yields a lower score or does not make any changes at all if $w^T_{y_i} x_i > w^T_j x_i$ by at least $\Delta$.
{: .prompt-info}

## Conclusion