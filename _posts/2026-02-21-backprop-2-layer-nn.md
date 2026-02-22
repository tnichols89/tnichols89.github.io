---
title: '2-Layer Neural Net Gradient Derivations for Backpropagation'
date: 2026-02-21 03:15:00 -0700
categories: [ML, Derivation]
tags: [backpropagation, classification]
math: true
---
# Overview
In this post, we derive the gradients required to update weight matrices and bias vectors during backpropagation through a simple 2-layer neural network. We outline the forward pass equations for our neural network before deriving the gradients that can be used directly in code to implement backpropagation and update the network to iteratively improve performance on the classification task at hand.

This 2-layer neural network was originally defined as part of a problem set from CS229 at Stanford University. In that context, this model was intended to be trained on the MNIST handwritten digit dataset for the purposes of identifying handwritten numerals from 0 through 9.

# Variables
Throughout this post, we will use the following variables when defining the dimensions of various matrices and vectors:
- $B$: batch size
- $n$: total number of samples
- $i$: index of a specific sample within the training data or index of a specific layer within the neural network
- $k$: number of classes; equal to 10 in the case of the MNIST handwritten digit dataset
- $h$: number of hidden dimensions in our network
- $d$: input data dimensionality; equal to the number of grayscale pixels in input images of handwritten digits, $28 \times 28$ in the case of the MNIST dataset

# Dimensions
- `training_data` $\in \mathbb{R}^{n \times d}$
  - The entire training data set
- $x^{(i)} \in \mathbb{R}^d$ is a
  - $d$-dimensional vector containing grayscale pixels with values 0 through 255 corresponding to a specific input image at index $i$ in `training_data`
- $y^{(i)} \in \mathbb{R}^k$
  - One-hot $k$-dimensional vector containing a $1$ at the index corresponding to the correct digit value from 0 through 9
- $\hat{y}^{(i)} \in \mathbb{R}^k$
  - $k$-dimensional vector containing the predicted probability distribution over the indices of the handwritten digits
- $\ell \in \mathbb{N}$
  - Index of $y$ and $\hat{y}$ corresponding to the correct ground truth label for a given input
- $a^{(i)} \in \mathbb{R}^h$
- $W^{[1]} \in \mathbb{R}^{d \times h}$
- $W^{[2]} \in \mathbb{R}^{h \times k}$
- $b^{[1]} \in \mathbb{R}^h$
- $b^{[2]} \in \mathbb{R}^h$
- $z^{(i)} \in \mathbb{R}^k$
- $Z \in \mathbb{R}^{B \times k}$


# Forward Pass
Since we are building a neural network for a classification task involving 10 classes, we begin with the cross entropy objective function as the root node of our network:
$$
\begin{align*}
  J &= \text{CrossEntropy}(\vec{y}, \hat{y}) & \text{where } \vec{y}, \hat{y} \in \mathbb{R}^{k \times 1} \\
    &= -\Sigma_{k=1}^k y_k \log(\hat{y}_k) \\
    &= -y_\ell \log(\hat{y}_{\ell}) & y_k = 0 \forall k \neq \ell\\
    &= -y_\ell \log(\text{softmax}_\ell(z^{[2]}))
\end{align*}
$$

And we define the "hidden" layers of our neural network as:
$$
\begin{align*}
  z^{[2]} &= W^{[2]T} a^{[2]} + b^{[2]} &\in \mathbb{R}^{k \times 1} \\
  a^{[1]} &= \sigma (z^{[1]}) &\in \mathbb{R}^{h \times 1} \\
  z^{[1]} &= W^{[1]T} x + b^{[1]} &\in \mathbb{R}^{h \times 1}
\end{align*}
$$

Where our activation function is defined as the typical sigmoid function:
$$
\sigma(x) = \frac{1}{1+e^{-x}} \forall x \in \mathbb{R}
$$

When we apply the sigmoid function to a vector, it acts as an element-wise operation.

# Gradients & Backpropagation
In order to implement backpropagation, we need to derive closed form mathematical expressions for the gradients of all the learnable parameters in our network. These gradients can be used in conjunction with various neural network optimization techniques from simple gradient descent methods to more advanced Adam-based methods to update the learnable parameters in the model after each forward pass.

Optimization techniques are out of scope of this post; we instead focusing on deriving the gradients required by optimization algorithms.

All of the learnable parameters in our network reside in $W^{[1]}$, $b^{[1]}$, $W^{[2]}$, and $b^{[2]}$ so we need to compute $\frac{\partial J}{\partial W^{[1]}}$, $\frac{\partial J}{\partial b^{[1]}}$, $\frac{\partial J}{\partial W^{[2]}}$, and $\frac{\partial J}{\partial b^{[2]}}$.

## Gradient Derivations
We begin by deriving the gradient of our cross entropy loss function with respect to the final output layer of our neural network, $z^{[2]}$

Note that we have [already derived the gradient of the cross entropy function]({% post_url 2023-10-18-cross-entropy %}) with respect to $\hat{y}$ and found that:
$$
\begin{align*}
  \frac{\partial J}{\partial z^{[2]}} &= \frac{\partial \text{CrossEntropy}(\vec{y}, \hat{y})}{\partial z^{[2]}} \\
  &= \hat{y} - \vec{y} \in \mathbb{R}^{k \times 1}
\end{align*}
$$

This will serve as the "entrypoint" into the rest of the recursive backpropagation process for the rest of our network.

We start by finding $\frac{\partial J}{\partial W^{[2]}}$ using the chain rule of calculus:
$$
\begin{align*}
  \frac{\partial J}{\partial W^{[2]}} &= \left[ \frac{\partial J}{\partial z^{[2]}} \cdot \left[ \frac{\partial z^{[2]}}{\partial W^{[2]}} \right] ^T \right] ^T \\
  &= \left[ \frac{\partial J}{z^{[2]}} \cdot a^{[1]T} \right] ^T \\
  &= a^{[1]} \cdot \left[ \frac{\partial J}{\partial z^{[2]}} \right] ^T \\
  &= a^{[1]} \cdot (\hat{y} - \vec{y})^T \in \mathbb{R}^{h \times k}
\end{align*}
$$

Where:
$$
\begin{align*}
  \frac{\partial J}{\partial z^{[2]}} &= (\hat{y} - \vec{y}) &\in \mathbb{R}^{k \times 1} \\
  \frac{\partial z^{[2]}}{\partial W^{[2]}} &= a^{[1]} &\in \mathbb{R}^{h \times 1} \\
  \frac{\partial J}{\partial W^{[2]}} &= a^{[1]} \cdot (\hat{y} - \vec{y})^T &\in \mathbb{R}^{h \times k}
\end{align*}
$$

---

Next, we find $\frac{\partial J}{\partial b^{[2]}}$, again using the chain rule:
$$
\begin{align*}
  \frac{\partial J}{\partial b^{[2]}} &= \left[ \frac{\partial J}{\partial z^{[2]}} \right] ^T \cdot \frac{\partial z^{[2]}}{\partial b^{[2]}} \\
  &= \left[ \frac{\partial J}{\partial z^{[2]}} \right] ^T \cdot \mathbb{1} \\
  &= \left[ \frac{\partial J}{\partial z^{[2]}} \right] ^T \\
  &= (\hat{y} - \vec{y})^T &\in \mathbb{R}^{1 \times k} \\
  &= \hat{y} - \vec{y} &\in \mathbb{R}^{k \times 1}
\end{align*}
$$

Where:
$$
\begin{align*}
  \frac{\partial J}{\partial z^{[2]}} &= (\hat{y} - \vec{y}) &\in \mathbb{R}^{k \times 1} \\
  \frac{\partial z^{[2]}}{\partial b^{[2]}} &= \mathbb{I} &\in \mathbb{R}^{k \times k}
\end{align*}
$$

This is a little mathematically sloppy with the transpositions. In machine learning, the convention is to ensure the shapes of our gradients align with the vector/matrix with respect to which we are differentiating the loss function so that we can apply those gradients to those matrices/vectors during the optimization step to update the parameters of our model.

Thus, we apply transposes liberally when needed to ensure our matrix and vector dimensions align properly.

---

We now move to the first layer of our neural network by finding $\frac{\partial J}{\partial W^{[1]}}$ using the chain rule, as usual:

$$
\begin{align*}
  \frac{\partial J}{\partial W^{[1]}} &= \left[ \left[ \left[ \left[ \frac{\partial J}{\partial z^{[2]}} \right]^T \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}} \right] ^T \odot \frac{\partial a^{[1]}}{\partial z^{[1]}} \right] \otimes \left[ \frac{\partial z^{[1]}}{\partial W^{[1]}} \right] ^T \right] ^T &\in \mathbb{R}^{d \times h} \\
\end{align*}
$$

Where $\odot$ denotes the Hadamard product and $\otimes$ denotes the outer product.

> We use the Hadamard product with the $\frac{\partial a^{[1]}}{\partial z^{[1]}}$ term since $a^{[1]}$ is an element-wise activation function and so the gradient flows through that layer on an element-wise basis.
{: .prompt-info}

> Using the outer product with the $\frac{\partial z^{[1]}}{\partial W^{[1]}}$ term allows us to construct the gradient for $W^{[1]}$ using two vectors with dimensions $h \times 1$ and $d \times 1$ rather than going through a 3-dimensional tensor, which is the longer and more computationally inefficient way to go about constructing the gradient.
{: .prompt-tip}

We already know that $\frac{\partial J}{\partial z^{[2]}} = (\hat{y} - \vec{y}) \in \mathbb{R}^{k \times 1}$ so we derive the $\frac{\partial z^{[2]}}{\partial a^{[1]}}$, $\frac{\partial a^{[1]}}{\partial z^{[1]}}$, and $\frac{\partial z^{[1]}}{\partial W^{[1]}}$ terms one at a time while remaining cognizant of the dimensionality of each term:
$$
\begin{align*}
  \frac{\partial J}{\partial z^{[2]}} &= (\hat{y} - \vec{y}) &\in \mathbb{R}^{k \times 1} \\
  \frac{\partial z^{[2]}}{\partial a^{[1]}} &= W^{[2]} &\in \mathbb{R}^{k \times h} \\
  \frac{\partial a^{[1]}}{\partial z^{[1]}} &= \sigma(z^{[1]}) \odot (1 - \sigma(z^{[1]})) \\
  &= a^{[1]} \odot (1 - a^{[1]}) &\in \mathbb{R}^{h \times 1} \\
  \frac{\partial z^{[1]}}{\partial W^{[1]}} &= \vec{x} &\in \mathbb{R}^{d \times 1}
\end{align*}
$$

> Notable caveat regarding $\frac{\partial z^{[1]}}{\partial W^{[1]}}$: In machine learning, we typically expect gradients to have the same shape as the vector/matrix with respect to which we're taking the derivative so we can use the gradient to update that vector/matrix. However, we can plainly see that isn't the case here since $W^{[1]}$ and $\vec{x}$ have different shapes. In the case of $z^{[1]}$, we can conceptually view $\vec{x}$ as influencing all rows of $W^{[1]}$, so we use that as a shortcut to construct the gradient of $W^{[1]}$ using the outer product of the upstream and local gradients.
{: .prompt-warning}

Finally, we combine all the terms piece by piece:
$$
\begin{align*}
  \frac{\partial J}{\partial z^{[2]}} &= (\hat{y} - \vec{y}) &\in \mathbb{R}^{k \times 1} \\
  \frac{\partial J}{\partial a^{[1]}} &= \left[ \frac{\partial J}{\partial z^{[2]}} \right]^T \cdot \frac{\partial z^{[2]}}{\partial a^{[1]}} \\
  &= (\hat{y} - \vec{y})^T \cdot W^{[2]T} &\in \mathbb{R}^{1 \times h} \\
  \frac{\partial J}{\partial z^{[1]}} &= \left[ \frac{\partial J}{\partial a^{[1]}} \right]^T \odot \frac{\partial a^{[1]}}{\partial z^{[1]}} \\
  &= \left[ (\hat{y} - \vec{y})^T \cdot W^{[2]T} \right]^T \odot a^{[1]} \odot (1-a^{[1]}) \\
  &= \left[ W^{[2]} \cdot (\hat{y} - \vec{y}) \right] \odot a^{[1]} \odot (1-a^{[1]}) &\in \mathbb{R}^{h \times 1}\\
  \frac{\partial J}{\partial W^{[1]}} &= \left[ \frac{\partial J}{\partial z^{[1]}} \otimes \left[ \frac{\partial z^{[1]}}{\partial W^{[1]}} \right]^T \right]^T \\
  &= \left[ \left[ \left[ W^{[2]} \cdot (\hat{y} - \vec{y}) \right] \odot a^{[1]} \odot (1-a^{[1]}) \right] \otimes\vec{x}^T \right]^T \\
  &= \vec{x} \otimes \left[ \left[ W^{[2]} \cdot (\hat{y} - \vec{y}) \right] \odot a^{[1]} \odot (1-a^{[1]}) \right]^T &\in \mathbb{R}^{d \times h}
\end{align*}
$$

---

All we have left to do now is derive $\frac{\partial J}{\partial b^{[1]}}$:
$$
\begin{align*}
  \frac{\partial J}{\partial b^{[1]}} &= \frac{\partial J}{\partial z^{[1]}} \cdot \frac{\partial z^{[1]}}{\partial b^{[1]}} \\
  &= \frac{\partial J}{\partial z^{[1]}} \cdot \mathbb{I} \\
  &= \frac{\partial J}{\partial z^{[1]}} \\
  &= \left[ W^{[2]} \cdot (\hat{y} - \vec{y}) \right] \odot a^{[1]} \odot (1-a^{[1]}) &\in \mathbb{R}^{h \times 1}
\end{align*}
$$

# Conclusion
We have found that:
$$
\begin{align*}
  \frac{\partial J}{\partial W^{[2]}} &= a^{[1]} \cdot (\hat{y} - \vec{y})^T &\in \mathbb{R}^{h \times k} \\
  \frac{\partial J}{\partial b^{[2]}} &= \hat{y} - \vec{y} &\in \mathbb{R}^{k \times 1} \\
  \frac{\partial J}{\partial W^{[1]}} &= \vec{x} \otimes \left[ \left[ W^{[2]} \cdot (\hat{y} - \vec{y}) \right] \odot a^{[1]} \odot (1-a^{[1]}) \right]^T &\in \mathbb{R}^{d \times h} \\
  \frac{\partial J}{\partial b^{[1]}} &= \left[ W^{[2]} \cdot (\hat{y} - \vec{y}) \right] \odot a^{[1]} \odot (1-a^{[1]}) &\in \mathbb{R}^{h \times 1}
\end{align*}
$$

These gradients can be directly computed and applied to our network to update the learnable parameters using gradient descent or other more sophisticated parameter optimization techniques such as AdamW.