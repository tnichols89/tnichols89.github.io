---
title: 'Learned Aggregation'
date: 2023-10-15 15:00:00 -0700
categories: [ML]
tags: [aggregation, attention]
math: true
---
## Overview
Vector aggregation is a common task in machine learning.

The general objective behind vector aggregation is: given a set of vectors with some shape $(\ell, d)$, we want to produce a single vector $(1, d)$ that represents the entire set of of $\ell$ vectors. Aggregating sets of vectors in this way provides a convenient means to represent internal state for downstream use akin to RNNs or to reduce the overall size of a state matrix - as with hidden states in transformers, for example - in a relatively efficient way.

In this post, we briefly explore vector aggregation operations and articulate a form of learned aggregation based on a slightly modified version of the self-attention mechanism used in transformers.

## Background
There are multiple common approaches to vector aggregation of varying complexity and popularity. Some are particularly suitable to certain types of problems.

For contexts in which there is no inherent ordering, we are limited to the following permutation-invariant approaches:
- Sum aggregation
- Mean aggregation
- Max aggregation

These approaches are common throughout many subfields of ML. They may be the *only* choice in some cases, though. For example, when we are applying neural networks to vertex-edge graphs to generate embeddings for nodes and edges based on neighboring nodes or adjacent edges, we must use a permutation-invariant approach in order to reflect the fact that graphs do not encode any cardinality between neighbors and edges. If the ordering of the neighbors has no semantic meaning in the graph, it should not have any semantic impact on our embeddings.

Other contexts are permutation-variant. Computer vision and natural language processing are examples of fields in which the ordering of input into the model is often semantically significant. We are free to choose a permutation-variant or permutation-invariant approach for such applications. For such applications, permutation-invariant approaches are simple and fast to implement *and* train but they are generally less expressive and therefore may yield inferior performance to permutation-variant approaches.

Permutation-variant approaches include:
- Linear projection
- Learned aggregation

With a linear projection, we essentially just flatten the output of a model and feed it forward as input to one or more densely-connected linear layers with an output dimensionality matching our desired output vector length.

Learned aggregation, however, is a more interesting approach based on a modified form of the self-attention mechanism used in transformers that may perform slightly better under some circumstances than a simple linear projection.

## Learned Aggregation
Given a feature matrix $X \in \mathbb{R}^{\ell \times d}$, learned aggregation can be formalized as:

$$
f(X) = \text{softmax}\left( \frac{\left(q_{cls} W^Q\right) \left(X^T W^K\right)}{\sqrt{d_k}} \right)(X^T W^V) \in \mathbb{R}^{1 \times p}
$$

with learnable parameters:
- $q_{cls} \in \mathbb{R}^{1 \times n}$
- $W^Q \in \mathbb{R}^{n \times d}$
- $W^K \in \mathbb{R}^{\ell \times d}$
- $W^V \in \mathbb{R}^{\ell \times p}$

where:
- $\ell$ is the cardinality (i.e. size) of the set of vectors we are aggregating
- $d$ is the embedding dimensionality of the input feature matrix
- $n$ is a hyperparameter influencing the dimensionality - and therefore expressiveness - of the query term
- $p$ is a hyperparameter representing the desired dimensionality of the output
  - The output vector will be of size $(1, p)$

> It may be helpful to contextualize $\ell$ and $d$ in terms of hidden states in a transformer, where you have a shape like $(b, \ell, d)$ with $b$ as the batch dimension, $\ell$ as the maximum sequence length, and $d$ as the token embedding dimensionality
{: .prompt-info}

## Code
An example of a learned aggregation model implemented using PyTorch is as follows:

```python
import torch
from torch import nn

class LearnedAggregation(nn.Module):
    def __init__(
        self,

        # Maximum sequence length of input: corresponds to l
        max_seq_len: int,

        # Dimensionality of embeddings in feature matrix X: corresponds to d
        hidden_dim: int,

        # Desired dimensionality of output vector: corresponds to p
        out_dim: int,

        # Desired dimensionality of query term: corresponds to n
        query_dim: int = 64,
    ) -> None:
        super().__init__()

        self.out_dim = out_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        self.q_cls = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(
                    1,
                    query_dim,
                    dtype=torch.float,
                )
            )
        )
        self.wq = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(
                    query_dim,
                    self.hidden_dim,
                    dtype=torch.float,
                )
            )
        )
        self.wk = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(
                    self.max_seq_len,
                    self.hidden_dim,
                    dtype=torch.float,
                )
            )
        )
        self.wv = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(
                    self.max_seq_len,
                    self.out_dim,
                    dtype=torch.float,
                )
            )
        )

        # We want the scale factor to be on the same device by default
        # so we make it a parameter. Since it's fixed, however, we
        # disable gradient for it.
        self.scale_factor = nn.Parameter(
            torch.sqrt(
                torch.tensor([self.hidden_dim], dtype=torch.float)
            ),
            requires_grad=False,
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        # X.size(): (b, l, d)

        intermediate: torch.tensor = torch.matmul( # => (b, 1, d)
            # Add batch dim to support broadcasting.
            #
            # PyTorch would handle this properly even if we didn't
            # add the batch dimension but we add it anyway to be
            # explicit.
            torch.unsqueeze( # (1, d) => (1, 1, d)
                torch.matmul( # => (1, d)
                    self.q_cls, # q_cls: (1, n)
                    self.wq # W^Q: (n, d)
                ),
                dim=0,
            ),

            torch.matmul( # => (b, d, d)
                torch.transpose(X, 1, 2), # X: (b, l, d) => (b, d, l)

                # Add the batch dimension again to be explicit.
                torch.unsqueeze( # (l, d) => (1, l, d)
                    self.wk, # W^K: (l, d)
                    dim=0,
                )
            )
        ) / self.scale_factor

        aggregated_vectors: torch.tensor = torch.matmul( # (b, 1, p)
            torch.softmax(intermediate, dim=2), # (b, 1, d)
            torch.matmul( # (b, d, p)
                torch.transpose(X, 1, 2), # X: (b, l, d) => (b, d, l)

                # Add the batch dimension again to be explicit.
                torch.unsqueeze( # (l, p) => (1, l, p)
                    self.wv, # W^V: (l, p)
                    dim=0,
                )
            )
        )

        # Return a batch of aggregated vectors. There is exactly one
        # vector per batch element, and each vector has dimensionality
        # p (self.out_dim).
        #
        # (b, 1, p) => (b, p)
        return torch.squeeze(aggregated_vectors)
```

## Performance
In problem spaces to which linear projection and learned aggregation are applicable - i.e. permutation-variant problem spaces - we can generally expect linear projection and learned aggregation to yield modest improvements over permutation-invariant approaches.

Between linear projection and learned aggregation, however, we generally expect similar performance. Anecdotally, there have been maybe one or two projects where I have empirically observed a marginal performance boost using learned aggregation compared to linear projection, but with a slightly higher implementation complexity.

Such marginal improvements may be attributable to the simple fact that learned aggregation generally introduces more parameters than a simple single-layer linear projection though, so you might be able to pull them even by simply adding more dense linear layers. Introducing some nonlinearity (e.g. ReLU activations) to both may also help boost performance further.

## Conclusion
In many cases, learned aggregation as defined and implemented above will perform approximately the same as simply flattening the feature matrix from $(b, \ell, d)$ to $(b, \ell * d)$ and following it up with a simple linear projection to the desired output dimensionality.

You may, however, find that a more sophisticated approach to learned aggregation can provide your model with notable benefits on your specific dataset. As is the case with most things in ML, different approaches may perform better or worse based on many different factors, including those specific to your dataset, and it may not always be clear why.

So, it's worth a shot! Might as well add another subtree to your model design decision tree, right?