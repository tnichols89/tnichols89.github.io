---
title: 'Learned Aggregation'
date: 2023-10-15 15:00:00 -0700
categories: [ML]
tags: [aggregation, attention]
math: true
---
## Overview
Vector aggregation is a common task in machine learning.

In subfields of ML that focus on applying neural networks to graphs, vector aggregation is a central component used to produce vectors for nodes and edges based on information from neighboring nodes and adjacent edges. In subfields of ML that use transformers for natural language processing or computer vision tasks, vector aggregation is sometimes used 

## Common Approaches
There are multiple common approaches to vector aggregation of varying complexity and popularity. Some are particularly suitable to certain types of problems.

For contexts in which there is no inherent ordering, we are limited to the following permutation-invariant approaches:
- Sum aggregation
- Mean aggregation
- Max aggregation

These approaches are common throughout many subfields of ML. They may be the *only* choice in some cases, though. For example, when we are applying neural networks to vertex-edge graphs to generate embeddings for nodes and edges based on neighboring nodes or adjacent edges, we must use a permutation-invariant approach in order to reflect the fact that graphs do not encode any cardinality between neighbors and edges. If the ordering of the neighbors has no semantic meaning in the graph, it should not have any semantic impact on our embeddings.

Other contexts are permutation-variant. Computer vision and natural language processing are examples of fields in which the ordering of input into the model is often semantically significant. We are free to choose a permutation-variant or permutation-invariant approach for such applications.

Permutation-variant approaches include:
- Linear projection
- Learned aggregation

With a linear project, we essentially just flatten the output of a model and feed it forward as input to one or more densely-connected linear layers with an output dimensionality matching our desired output vector length.

Learned aggregation, however, is a more interesting approach that may perform slightly better under some circumstances than a simple linear projection.

## Goal
In this post, we introduce a form of learned aggregation as a slightly modified version of the self-attention mechanism used in transformers.

## Learned Aggregation

$$
\text{softmax}\left( \frac{\left(q_{cls} Q\right) \left(X^T K\right)}{d_k} \right)(X^T V) \in \mathbb{R}^{1 \times p}
$$

where the feature matrix $X \in \mathbb{R}^{\ell \times d}$, with learnable parameters:
- $q_{cls} \in \mathbb{R}^{1 \times n}$
- $Q \in \mathbb{R}^{n \times d}$
- $K \in \mathbb{R}^{\ell \times d}$
- $V \in \mathbb{R}^{\ell \times p}$


## Performance
In problem spaces to which linear projection and learned aggregation are applicable - i.e. permutation-variant problem spaces - we can generally expect linear projection and learned aggregation to yield moderate improvements over permutation-invariant approaches.

Between linear projection and learned aggregation, however, we generally expect similar performance. Anecdotally, there have been maybe one or two projects where I have empirically observed a marginal performance boost using learned aggregation compared to linear projection, but with a slightly higher implementation complexity.

Such marginal improvements may be attributable to the simple fact that learned aggregation generally introduces more parameters than a simple linear projection using one or two dense linear layers though, so you might be able to pull them even by simply adding more dense linear layers. Introducing some nonlinearity (e.g. ReLU activations) to both may also help boost performance further.