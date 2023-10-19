---
title: 'SVMs and the Hinge Loss Function'
date: 2023-10-19 11:30:00 -0700
categories: [ML, Derivation]
tags: [loss function, classification, supervised training]
math: true
---
## Overview
Support vector machines (SVMs) are arguably the most powerful linear classification models in widespread use. They are linear models that can be trained using the hinge loss function, among other similar linear loss functions.

SVMs work very well in part because the hinge loss function - a loss function commonly used by SVMs - incentivizes the model to generate high scores for the correct classification of a given sample while *simultaneously* pushing scores corresponding to incorrect classification lower than that of the correct classification by a hyperparameterized margin $\Delta$.

The margin concept may seem a bit abstract at first, but its utility becomes more apparent as we dive into the definition of the hinge loss function used by SVMs.

SVMs can also handle linearly separable data when trained with the hinge loss function, and the convexity of the hinge loss function means that the model reliably converges during training.

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

is referred to as the "indicator function." It simply evaluates to the number to the left ($\mathbb{1} in this case$) when the `<conditional>` inside of the curly braces evaluates to true. This notation is often used for counting in ML texts.

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

These two gradients give us everything we need to update both of the learnable parameters used by the hinge loss function to train our support vector machine.

#### Disappearing Sum
You may have noticed that one of our gradients retains the summation from the original loss function and the other does not. If you already have a solid understanding as to why that is, feel free to skip this section. If not, we encourage you to keep reading as it is a critically important component of the gradients.

Recall that the original loss function consists of a sum over $\bigm\| K \bigm\| - 1$ terms, where $\bigm\| K \bigm\|$ is the total number of possible classifications for any given sample:

$$
\ell(x_i) = \Sigma_{\substack{j \in K \\ j \neq y_i}} \begin{cases}
  w^T_j x_i - w^T_{y_i} x_i + \Delta \quad & \text{if } w^T_j x_i - w^T_{y_i} x_i + \Delta > 0 \\
  0 \quad & \text{otherwise}
\end{cases}
$$

This sum iterates over $j \in K$ while excluding the one term in which $j = y_i$. In a scenario where we have 10 possible classifications, this sum will contain 9 terms. However, no matter how many terms there are, **every single term in the sum still contains $- w^T_{y_i} x_i$**.

The gradient of our loss function $\ell$ with respect to $w_{y_i}$ is consequently a sum over the derivative of each and every one of the terms contained in the original loss function.

Thus, the gradient of our loss function $\ell$ with respect to $w_{y_i}$ retains the exact same sum:

$$
  \frac{\partial \ell(x_i)}{\partial w_{y_i}} = \Sigma_{\substack{j \in K \\ j \neq y_i}} \begin{cases}
    -x_i \quad & \text{if } w^T_j x_i - w^T_{y_i} x_i + \Delta > 0 \\
    0 \quad & \text{otherwise}
  \end{cases}
$$

This stands in contrast to the gradient of our loss function $\ell$ with respect to $w_j$:

$$
\frac{\partial \ell(x_i)}{\partial w_{j}} = \begin{cases}
  x_i \quad & \text{if } w^T_j x_i - w^T_{y_i} x_i + \Delta > 0 \\
  0 \quad & \text{otherwise}
\end{cases}
$$

When we derive the gradient of our loss function $\ell$ with respect to a specific $w_j$, we are looking at the gradient of a specific weight vector $w_j$ corresponding to **one specific (incorrect) possible classification** of the given sample $x_i$, **not** all possible (incorrect) classifications.

For any given specific label $j$, the sum contained in our loss function $\ell$ will include precisely 1 term containing the corresponding $w_j$. When we take the gradient with respect to the weight vector $w_j$ of that specific label $j$, all but one of the terms in the sum becomes 0.

> On a more intuitive level, the gradient with respect to $w_{y_i}$ contains a sum over all the same terms as the original loss function because the loss function is designed to yield a score for the one correct label that is greater than the scores of **every** incorrect label. When updating the weight vector $w_j$ for a specific incorrect label $j$, however, we only care about making the score corresponding to *that one specific incorrect label* lower than the score of the correct label by at least $\Delta$.
{: .prompt-tip}

## Conclusion
The hinge loss function is an incredibly powerful tool for training support vector machines (SVMs).

Due to the simultaneous action of building up scores for correct labels and reducing scores for incorrect labels while *also* building in a margin term to encourage even more robust dispersal of scores, SVMs often yield models that are remarkably capable for many applications even when compared to their more modern neural network counterparts.

Moreover, the hinge loss function being as shallow as it is means that it trains incredibly fast. Modern laptop processors can train an SVM using hinge loss in milliseconds. The hinge loss function has the added advantage of convexity so it is capable of reliably converging, unlike most modern neural network architectures. 