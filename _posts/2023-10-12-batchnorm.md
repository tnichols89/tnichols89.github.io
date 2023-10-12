---
title: BatchNorm Backpropagation Derivation
date: 2023-10-11 20:00:00 -0800
categories: [AI, ML]
tags: [backpropagation, derivation]
math: true
---
# Overview

$$
\begin{align*}
  \mu_B &= \frac{1}{m} \Sigma^m_{i=1} x_i \\
  \sigma^2_B &= \frac{1}{m}\Sigma^m_{i=1}(x_i - \mu_B)^2 \\
  \hat{x}_i &= \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}} \\
  y_i &= \gamma \hat{x}_i + \beta
\end{align*}
$$

$$
$$
