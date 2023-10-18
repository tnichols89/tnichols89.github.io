---
title: 'Contrastive Learning'
date: 2023-10-17 12:00:00 -0700
categories: [ML, Training]
tags: [self-supervised training, supervised training, semi-supervised training]
math: true
---
## Overview
Contrastive learning is an approach to training neural networks using a training objective that differs in some key ways from other common training objectives.

Contrastive learning can be viewed as a less direct objective than many discriminative objectives like binary cross entropy, where the model is clearly incentivized to predict the correct label or labels for a given sample. Where discriminative objectives incentivize the model to behave in intuitive ways, contrastive loss can be a bit less intuitive.

Just like there are different loss functions that are useful for discriminative training objectives - softmax-based negative log likelihood, binary cross entropy, multi-class cross entropy, etc. - there are different loss functions we can use to implement contrastive learning.

In this post, we will explore contrastive learning, how it differs from purely discriminative objectives, and the advantages and disadvantages.

> Keep in mind that contrastive learning and discriminative learning are **not** mutually exclusive. They can be and often are combined to improve model performance as much as possible, especially in cases where the goal is to train a core model (or a "backbone" model) that we want to combine with numerous downstream task-specific models.
{: .prompt-info}

## Intuition
Contrastive learning is fundamentally not about classifying samples; it's about encouraging healthy distribution of embeddings in embedding space. The overarching concept of contrastive learning is simple: we use loss functions that incentivize the model to produce embeddings that are close to each other in embedding space for similar samples and embeddings that are far away from each other in embedding space for dissimilar samples.

The loss functions we use for contrastive learning are not at all parameterized with labels in the way that loss functions for discriminative objectives like cross entropy are; they are parameterized exclusively by embeddings produced by the model. We will often use sample labels to distinguish "similar" and "dissimilar" samples but we do not have to, and the loss function does not consume the labels directly. This is why contrastive learning approaches are useful for supervised, self-supervised, and semi-supervised training.

By training our model to produce and distribute embeddings in a certain way rather than training it to directly discriminate between similar or dissimilar samples, we often end up with embeddings that are more generally insightful for various downstream tasks. Models that are trained to produce embeddings distributed in a certain way throughout embedding space by using contrastive learning tend to produce embeddings that are more broadly useful for a larger range of downstream tasks.

It is for this reason that models trained using contrastive learning are so useful as a "core" or "backbone" model that consumes input, produces embeddings, and makes those embeddings available for consumption by task-specific, downstream models. This architectural approach can also save on training costs since the backbone model can be larger and more robust but only needs to be trained once before it is capable of supporting many downstream prediction tasks, and the task-specific downstream models consuming the embeddings produced by the backbone can often have far fewer parameters.

## Loss Function Preliminaries
We need to define what it means for embeddings to be "close" or "far away" and what it means for samples to be "similar" or "dissimilar" before moving forward.

### Embedding Distance
Contrastive learning approaches are fundamentally built upon embedding space distance metrics. Euclidean distance and cosine distance are two of the most commonly used metrics for measuring distance in embedding space, so we discuss both and explore their differences.

Let $\vec{p}, \vec{q} \in \mathbb{R}^n$.

#### Euclidean Distance
Euclidean distance is fairly easy to understand as it first occurs for most of us early in our algebra/geometry education.

Euclidean distance is formally defined as:

$$
d_e(\vec{p}, \vec{q}) = \sqrt{\Sigma^n_{i=1} (q_i - p_i)^2}
$$

Where $d_e(\vec{p}, \vec{q}) \in [0, \infty)$.

#### Cosine Distance
Cosine distance is a bit more interesting. Although cosine distance isn't a *true* distance metric because it does not maintain the Schwarz inequality, it still generally works well for the purposes of contrastive learning.

Let $\langle \vec{p}, \vec{q} \rangle$ denote the vector inner product. We first define cosine similarity $S_c(\vec{p}, \vec{q})$ as:

$$
\begin{align*}
  S_c(\vec{p}, \vec{q}) &= \frac{\langle \vec{p}, \vec{q} \rangle}{\lVert \vec{p} \rVert \cdot \lVert \vec{q} \rVert} \\
  &= \frac{\Sigma^n_{i=1} p_i q_i}{\sqrt{\Sigma^n_{i=1}p^2_i} \cdot \sqrt{\Sigma^n_{i=1}q^2_i}}
\end{align*}
$$

Where $S_c(\vec{p}, \vec{q}) \in [-1, 1]$.

We then define cosine distance as:

$$
d_c(\vec{p}, \vec{q}) = 1 - S_c(\vec{p}, \vec{q})
$$

Where $d_c(\vec{p}, \vec{q}) \in [0, 2]$.

#### Euclidean vs Cosine Distance
There is one functionally critical (for our purposes) difference between these two distance metrics: Euclidean distance captures differences in both angle and magnitude of $\vec{p}$ and $\vec{q}$ while cosine distance only captures difference in the angle between the two vectors.

This difference may or may not play a significant role in which metric is the best choice for a given model architecture - it requires consideration on a case-by-case basis. Euclidean distance does not include a vector magnitude normalization term as cosine distance does, and so it is capable of differentiating between vectors pointing in exactly the same direction but with different magnitudes. Cosine distance is unable to differentiate between two vectors based on their magnitudes alone.

A critically important contextual consideration: most model architectures normalize layer outputs using something like LayerNorm or BatchNorm. These normalization schemes normalize data to have 0 mean and unit variance. Where LayerNorm normalizes feature vectors to have 0 mean and unit variance independently of other feature vectors, BatchNorm normalizes feature vectors on a per-feature basis across all feature vectors in a batch. One normalization scheme may therefore yield feature vectors with higher-variance magnitudes than those feature vectors normalized using the other scheme.

> Intuitively, this critical difference between Euclidean and cosine distance is unlikely to precipitate substantial differences during training when the model architecture in question normalizes the embeddings it outputs to have 0 mean and unit variance. The differences may become more pronounced when using either LayerNorm or BatchNorm, and may be particularly apparent if the model architecture utilizes little to no normalization at all.
{: .prompt-tip}

Another consideration is the difference in images/ranges of the two distance metrics.

For the Euclidean distance metric, two identical vectors will have a distance of $0$ but two dissimilar vectors can have arbitrarily high distance.

For the cosine distance metric, two identical vectors will similarly have a distance of $0$ but two maximally dissimilar vectors can have at most a distance of $2$. This is because $S_c(\vec{p}, \vec{q}) = -1$ when $\vec{p}$ and $\vec{q}$ point in opposite directions, $S_c(\vec{p}, \vec{q}) = 0$ when $\vec{p}$ and $\vec{q}$ are orthogonal to each other, and $S_c(\vec{p}, \vec{q}) = -1$ when $\vec{p}$ and $\vec{q}$ point in the same exact direction.

This difference in maximum distance between the two metrics influences our choice of the margin hyperparameter, which we introduce when we define the loss functions themselves later on.

### Similar and Dissimilar Samples
How we define "similar" and "dissimilar" is dependent on whether our training is supervised, self-supervised, or semi-supervised, and is also dependent on the nature of our data and problem space.

In a supervised context, "similar" samples can simply be defined as samples with identical or nearly identical labels. "Dissimilar" samples can be similarly defined as samples with different labels. This is easy to reason about with discrete labels but can be a bit trickier for continuous labels. We may have to define a maximum delta above or below the continuous ground truth label for a given sample, outside of which, we consider other samples to be "similar" or "dissimilar".

In a self-supervised context, "similar" and "dissimilar" become even more context-dependent. A computer vision model, for example, may be able to construct "similar" samples by applying different random perturbations or modifications to a single ground truth image. It may similarly construct "dissimilar" samples by applying randomy perturbations or modifications to two different ground truth images.

Semi-supervised training pipelines will, of course, utilize some combination of both supervised and self-supervised approaches.

In my experience, defining and collating similar and dissimilar samples is by far the biggest time sink when implementing contrastive learning. My data preprocessing and sampling logic often ends up substantially longer and more complicated than the core training logic and the loss function implementation.

## Loss Function Definitions
Cosine embedding loss and triplet loss are two of the most common contrastive learning objectives in use today. We define and discuss both but implement only one.

### Cosine Embedding Loss
An implementation of [cosine embedding loss](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html) is provided by PyTorch, making it a very convenient choice. We note, however, that it is parameterized only by two vectors and a parameter indicating whether they correspond to similar or dissimilar samples.

> This formulation of cosine embedding loss is only capable of either coalescing embedding vectors from two similar samples or dispersing embedding vectors from two dissimilar samples in each pass. It does not do both simultaneously, as our triplet loss implementation below does.
{: .prompt-info}

Cosine embedding loss can be formalized as:

$$
\begin{equation*}
\ell(\vec{p}, \vec{q}, y) = \begin{cases}
  1 - S_c(\vec{p}, \vec{q}) \quad & \text{if } y = 1 \\
  \text{max}(0, S_c(\vec{p}, \vec{q}) - \text{margin}) \quad & \text{if } y = -1
\end{cases}
\end{equation*}
$$

We can build an intuitive understanding of how this loss function works by deconstructing both cases.

Before proceeding, note that $S_c(\vec{p}, \vec{q}) = -1$ when $\vec{p}$ and $\vec{q}$ point in opposite directions, $S_c(\vec{p}, \vec{q}) = 0$ when $\vec{p}$ and $\vec{q}$ are orthogonal, and $S_c(\vec{p}, \vec{q}) = 1$ when $\vec{p}$ and $\vec{q}$ point in the same exact direction. 

#### Case 1
When $y = 1$ for some given $\vec{p}$ and $\vec{q}$ - that is, when the two embedding vectors correspond to *similar* samples - we see that:

$$
\ell(\vec{p}, \vec{q}, y) = 1 - S_c(\vec{p}, \vec{q})
$$

If $\vec{p}$ and $\vec{q}$ are maximally dissimilar but correspond to *similar* samples, our loss becomes:

$$
\begin{align*}
  \ell(\vec{p}, \vec{q}, y) &= 1 - S_c(\vec{p}, \vec{q}) \\
  &= 1 - (-1) \\
  &= 2
\end{align*}
$$

Which is the maximum possible loss corresponding to the harshest possible penalty for clustering the embeddings perfectly incorrectly.

If, on the other hand, $\vec{p}$ and $\vec{q}$ are maximally similar *and* correspond to *similar* samples, our loss becomes:

$$
\begin{align*}
  \ell(\vec{p}, \vec{q}, y) &= 1 - S_c(\vec{p}, \vec{q}) \\
  &= 1 - 1 \\
  &= 0
\end{align*}
$$

Which is the lowest possible loss, yielding no changes to the model during backpropagation due to an upstream gradient of 0 originating from the root node (e.g. loss function) in the directed computational graph for the model. In this case, our model has achieved exactly what we wanted it to and so there are no updates to be made to the model parameters.  

#### Case 2
When $y = -1$ for some given $\vec{p}$ and $\vec{q}$ - that is, when the two embedding vectors correspond to *dissimilar* samples - we see that:

$$
\ell(\vec{p}, \vec{q}, y) = \text{max}(0, S_c(\vec{p}, \vec{q}) - \text{margin})
$$

The `margin` term here is a hyperparameter that we choose prior to training. This hyperparameter incentivizes embeddings corresponding to dissimilar examples to disperse by *at least* `margin` amount. Let $\text{margin} = 0.5$ for the sake of discussion.

If $\vec{p}$ and $\vec{q}$ are maximally dissimilar *and* correspond to *dissimilar* samples, our loss becomes:

$$
\begin{align*}
  \ell(\vec{p}, \vec{q}, y) &= \text{max}(0, S_c(\vec{p}, \vec{q}) - \text{margin}) \\
  &= \text{max}(0, -1 - 0.5) \\
  &= \text{max}(0, -1.5) \\
  &= 0
\end{align*}
$$

Which is the lowest possible loss, yielding no changes to the model during backpropagation due to an upstream gradient of 0 originating from the root node (e.g. loss function) in the directed computational graph for the model. In this case, our model has achieved exactly what we wanted it to and so there are no updates to be made to the model parameters. 

If, on the other hand, $\vec{p}$ and $\vec{q}$ are maximally similar *but* correspond to *dissimilar* samples, our loss becomes:

$$
\begin{align*}
  \ell(\vec{p}, \vec{q}, y) &= \text{max}(0, S_c(\vec{p}, \vec{q}) - \text{margin}) \\
  &= \text{max}(0, 1 - 0.5) \\
  &= \text{max}(0, 0.5) \\
  &= 0.5
\end{align*}
$$

Which is the maximum possible loss (considering our chosen `margin` hyperparameter value) corresponding to the harshest possible penalty for clustering the embeddings perfectly incorrectly.

> Intuitively, if two dissimilar examples have a cosine similarity greater than `margin`, we consider that a perfect-enough distribution of embeddings in embedding space and our loss becomes $0$. If two dissimilar examples have a cosine similarity less than `margin`, our loss (i.e. penalty) linearly scales up based on how close the two embeddings are in embedding space.
{: .prompt-tip}

### Triplet Loss
Triplet loss differs from cosine embedding loss in several key ways. Rather than being parameterized by only two vectors, triplet loss is parameterized by three vectors: an *anchor* vector, a *positive* vector, and a *negative* vector. The anchor vector is an embedding corresponding to any random sample, the positive vector is an embedding corresponding to a similar sample, and the negative vector is an embedding corresponding to a dissimilar sample.

Although PyTorch does not provide a native implementation of triplet loss, it is very easy to implement ourselves, as shown below.

Let $\vec{r}_a, \vec{r}_p, \vec{r}_n \in \mathbb{R}^n$ correspond to the $n$-dimensional anchor embedding, positive embedding, and negative embedding, respectively. Let $d$ be a vector-valued, embedding-space distance metric $d \colon \mathbb{R}^n \rightarrow \mathbb{R}$. Cosine distance and Euclidean distance are both viable options for $d$.

Triplet loss can then be formalized as:

$$
\ell(\vec{r}_a, \vec{r}_p, \vec{r}_n) = \text{max}(0, d(\vec{r}_a, \vec{r}_p) - d(\vec{r}_a, \vec{r}_n) + \text{margin})
$$

To build an intuitive understanding of how this loss function works, keep three things in mind:
- We want the distance between the anchor embedding and its corresponding positive sample embedding - $d(\vec{r}_a, \vec{r}_p)$ - to be as small as possible
- We want the distance between the anchor embedding and its corresponding negative sample embedding - $d(\vec{r}_a, \vec{r}_n)$ - to be as large as possible
- Ideally, $d(\vec{r}_a, \vec{r}_p)$ would always be greater than $d(\vec{r}_a, \vec{r}_n)$ by at least $\text{margin}$, in which case, we would achieve $0$ loss

Observe that triplet loss is not defined as a piecewise function - it combines all three vectors into one loss computation, and includes both positive and negative sample embeddings. The result is that triplet loss *simultaneously* disperses dissimilar vectors while coalescing similar vectors in embedding space.

## Code
With all the concepts and theory out of the way, we (finally) get to the triplet loss implementation. We have decided to use cosine distance as our embedding space distance metric but Euclidean distance can be used as a drop-in replacement with minimal modification of the business logic if that is more suitable for your model and data.

> If our model does not normalize embeddings, we are likely sacrificing performance in choosing cosine distance over Euclidean distance as our distance metric. If embeddings are normalized, cosine distance and Euclidean distance will often perform similarly well. If embeddings are not normalized and may therefore differ significantly in their magnitudes, Euclidean distance will be substantially more capable of capturing differences between vectors than cosine distance.
{: .prompt-warning}

```python
import torch
from torch import nn

class TripletLoss(nn.Module):
    def __init__(
        self,

        # Good choices of margin values are often in (0.0, 0.5]
        margin: float = 0.35
    ) -> None:
        super().__init__()

        self.cos_sim = nn.CosineSimilarity()
        self.margin = margin

    def forward(
        self,

        # All three parameters are expected to be of shape: (b, d)
        # with `b` as the batch size and `d` as the embedding dimension
        anchor: torch.tensor,
        positive: torch.tensor,
        negative: torch.tensor,
    ) -> torch.tensor:
        # Compute cosine distances for positive and negative sample embeddings
        #
        # Shape: (b, d) => (b)
        pos_dist: torch.tensor = 1 - self.cos_sim(anchor, positive)
        neg_dist: torch.tensor = 1 - self.cos_sim(anchor, negative)

        # Compute loss distances
        #
        # Shape: (b)
        dists: torch.tensor = pos_dist - neg_dist + self.margin

        return torch.mean( # Shape: (1)
            torch.maximum( # Shape: (b)
                dists,
                torch.zeros_like(dists),
            )
        )
```

## Conclusion
Both triplet loss and cosine embedding loss functions are great options for contrastive learning. I would expect to see similar performance between the two in most cases but, as in all things in ML, one may be more suitable for your model architecture, data, and objective than the other.

When using triplet loss, both Euclidean distance and cosine distance are generally sound options but it is imperative to keep in mind the differences between the two so you can choose whichever one suits your model architecture and data most effectively.