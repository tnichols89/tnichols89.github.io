---
title: 'Negative Log Likelihood and Softmax'
date: 2023-10-20 14:30:00 -0700
categories: [ML, Derivation]
tags: [loss function, classification, supervised training]
math: true
---
## Overview
Softmax is an ubiquitous function in modern machine learning. It is most often found as a top level component of classification loss functions like cross entropy and negative log likelihood. The softmax function itself both consumes and produces vectors, with the output vector having the same dimensionality as the input vector. The output vector is a representation of the input vector with some of the characteristics of a probability distribution.

That is to say, the elements of the output sum to approximately 1 but it is not, strictly speaking, a probability distribution at its most extreme ends. Because softmax relies on exponentiation, it can never analytically take on values exactly equal to 0 or 1 - it can only asymptotically approach them. In practice, numerical instability at the extremes are approximated with 0 or 1.

Nonetheless, the output from softmax is pretty much always interpreted as a probability distribution and it is from its probability-distribution-like characteristics that the utility of the softmax function grows. It converts logits produced by machine learning models into de facto probability distributions.

Softmax is a key top-level component of large language models in widespread use today. In many cases, one of the final steps that generative language models take is to compute a probability distribution over their entire vocabulary in order to determine which token should be generated next.

Today, we will be examining the softmax function in the context of negative log likelihood. Negative log likelihood is a loss function based on the softmax function that enables us to quite directly use softmax for classification tasks in which a given sample is predicted to have precisely one label from among a set of more than one possible labels.

> For situations in which a given sample may assume more than one label, we instead apply the sigmoid function to each output logit independently. Multi-classification is out of the scope of this article.
{: .prompt-info}

## NLL and Softmax
### Definition
The softmax function is defined as:

$$
\text{softmax}(x_i) = \frac{\exp\left\{f_{y_i}(x_i)\right\}}{\Sigma^k_j \exp\left\{f_{j}(x_i)\right\}}
$$

The negative log likelihood loss function is defined as:

$$
\begin{align*}
  \ell(x_i) &= -\log(\hat{y}(x_i))\\
  &= -\log \left( \frac{\exp\left\{f_{y_i}(x_i)\right\}}{\Sigma^k_j \exp\left\{f_{j}(x_i)\right\}} \right) \\
  % &= -\log \left( \frac{\exp\left\{f_{y_i}\right\}}{\Sigma^k_j \exp\left\{f_{j}\right\}} \right) \\
  &= -\left[ \log\left( \exp\left\{ f_{y_i}(x_i) \right\} \right) - \log\left( \Sigma^k_j \exp\left\{f_{j}(x_i)\right\} \right)\right] \\
  &= -f_{y_i}(x_i) + \log\left( \Sigma^k_j \exp\left\{f_{j}(x_i)\right\} \right) \\
  &= - \left[ W x_i \right]_{y_i} + \log\left( \Sigma^k_j \exp\left\{\left[ W x_i \right]_j\right\} \right)
\end{align*}
$$

Where:
- $\hat{y}$ is a likelihood function, defined as $\hat{y}(x_i) = \text{softmax}(x_i)$ in most cases
- $k \in \mathbb{R}$ is the number of classes over which we want to generate a probability distribution
- $x_i \in \mathbb{R}^d$ is a $d$-dimensional feature vector corresponding to some given sample $i$
- $W \in \mathbb{R}^{k \times d}$ is a learnable weight matrix
- $f(x_i) = W x_i \in \mathbb{R}^k$ is a score function producing the logits over which we generate the probability distribution
  - This can be - and often is - replaced with a more sophisticated, deeper neural network
- $f_j(x_i) = \left[ W x_i \right]_j \in \mathbb{R}$ is the score corresponding to a specific possible class label $j$ for sample $x_i$
- $f_{y_i}(x_i) = \left[ W x_i \right]_{y_i}\in \mathbb{R}$ is the score corresponding to the correct class label $y_i$ for sample $x_i$

> Intuitively, the higher the score for the correct label $f_{y_i}(x_i)$, the lower the loss. The $\Sigma^k_j \exp\left[f_{j}(x_i)\right\]$ in the denominator simply creates an upper bound on the function at $1.0$. The numerator and denominator both contain exponentials to prevent either of them from ever becoming negative.
{: .prompt-tip}

### Numerical Stability
The exponentials in softmax directly provide the guarantee that the output values cannot drop below $0$ or exceed $1$. However, exponentials can quickly introduce numerical instability on computers due to overflow and underflow when numbers become very large or very small, respectively. There is a [fantastic discussion on Stack Overflow](https://stackoverflow.com/questions/42599498/numerically-stable-softmax#answer-49212689) regarding this behavior and how to mitigate it.

Using simple algebra - specifically, properties of exponents - we can effectively mitigate overflow and underflow issues in softmax:

```python
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    z: np.ndarray = x - np.max(x, axis=-1, keepdims=True)
    numerator: np.ndarray = np.exp(z)
    denominator: np.ndarray = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator
```

### Gradients

By the chain rule of calculus:

$$
\frac{\partial \ell}{\partial W} = \frac{\partial \ell}{\partial f} \frac{\partial f}{\partial W}
$$

Starting with the leftmost term in our chain rule equation:

$$
\begin{align*}
  \frac{\partial \ell(x_i)}{\partial f_p(x_i)} &= \begin{cases}
    -1 + \frac{1}{\Sigma^k_j \exp\left\{ f_j(x_i) \right\}} \cdot \exp\left\{ f_p(x_i) \right\} \quad & \text{if } p = y_i \\
    \frac{1}{\Sigma^k_j \exp\left\{ f_j(x_i) \right\}} \cdot \exp\left\{ f_p(x_i) \right\} \quad & \text{otherwise} 
  \end{cases} \\
  &= \begin{cases}
    -1 + \frac{\exp\left\{ f_p(x_i) \right\}}{\Sigma^k_j \exp\left\{ f_j(x_i) \right\}} \quad & \text{if } p = y_i \\
    \frac{\exp\left\{ f_p(x_i) \right\}}{\Sigma^k_j \exp\left\{ f_j(x_i) \right\}} \quad & \text{otherwise} 
  \end{cases}
\end{align*}
$$

Before proceeding with the rightmost term in our chain rule equation, first observe that:

$$
\frac{\partial f}{\partial W_p} = \frac{\partial}{\partial W_p}\left[ W_p x_i \right] = x_i
$$

Where $W_p$ denotes a single row in $W$ and $W_p x_i = \langle W_p, x_i \rangle \in \mathbb{R}$ is the dot product representing the score for classifcation label $p$ relative to sample feature vector $x_i$ for sample $i$.

Then:

$$
\frac{\partial \ell(x_i)}{\partial W_p} = \begin{cases}
  \left( \frac{\exp\left\{ f_p(x_i) \right\}}{\Sigma^k_j \exp\left\{ f_j(x_i) \right\}} - 1\right) x_i \quad & \text{if } p = y_i \\
  \left( \frac{\exp\left\{ f_p(x_i) \right\}}{\Sigma^k_j \exp\left\{ f_j(x_i) \right\}}\right) x_i \quad & \text{otherwise} 
\end{cases}
$$

### NumPy Code
The code to implement these gradients is definitely a bit more complicated than the derivation. The derivation is written in terms of individual $x_i$ samples and only "updates" one row $W_p$ at a time whereas the code must reckon with sample feature vectors in batch and yield all rows for the gradient of $W$ simultaneously to be efficient.

Anywhere in the derivation we see $W x_i$, we generally replace in code with a matrix multiplication $X W^T$ which is interpretable as a batch dot product operation for code efficiency.

Carefully note the differences in dimensions in the function preamble below relative to the dimensions used in the derivation above.

```python
import numpy as np
def softmax_with_gradients(
    W: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    reg: float
) -> tuple[float, np.ndarray]:
    """
    Dimensions:
    - b: batch size
    - d: feature vector dimensionality
    - c: number of classes

    Input shapes:
    - W: (d, c)
    - X: (b, d)
    - y: (b,)
        - y[i] = c means X[i] corresponds to label c, where 0 <= c < C
    - reg: regularization term

    Returns:
    - loss
    - gradient with respect to W
    """
    loss = 0.0
    dW = np.zeros_like(W)
    batch_size, _ = X.shape

    #
    # Compute the numerically stabilized softmax
    #
    # np.matmul((b, d), (d, c)) => (b, c)
    preds = np.matmul(X, W)
    adjusted_preds = np.exp(preds - np.amax(preds))

    # Denom must be an explicit column vector for broadcasting to work
    denom = np.sum(adjusted_preds, axis=1).reshape(-1, 1)
    proba = adjusted_preds / denom

    #
    # Compute the loss
    #
    loss -= np.sum(
        np.log(
            # An application of advanced integer array indexing in NumPy:
            # https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing
            #
            # This line selects the probability generated for the correct
            # class for each row in the batch. By passing in an integer
            # range to index the rows and a list `y` to index the columns,
            # NumPy uses the values stored within in the `y` array as
            # column indices.
            #
            # len(batch_size) must equal len(y) for this to work since the
            # column indices stored in `y` are associated with the batch
            # elements by their index in `y`.
            proba[range(batch_size), y]
        )
    )
    loss /= batch_size
    loss += reg * np.sum(W*W)

    #
    # Compute the gradient with respect to W
    #
    proba[range(batch_size), y] -= 1

    # np.dot((d, b),  (b, c)) => (d, c)
    dW = np.dot(X.T, proba)
    dW /= batch_size
    dW += 2 * reg * W

    return loss, dW
```

## Conclusion
The negative log likelihood loss function and the softmax function are natural companions and frequently go hand-in-hand.

This combination is the gold standard loss function for classification problems in which a model is required to produce a prediction for exactly one label per input sample. However, softmax itself is versatile enough that it is commonly used in other contexts as well.