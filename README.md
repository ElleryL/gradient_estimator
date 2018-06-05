# gradient_estimator

This algorithm are focused on how to efficiently estimate an gradient.

The gradient estimator is important in application as in many cases, we try to optimize (min/max) our objective function use gradient descent method.

However, we might encounter some challenges 
(1) unbiased estimator. 
(2) High Variances. 
(3) Discrete random variable (can't use reparameterization trick directly) 
(4) Expected Function un-differentiable

In this project, we aim to solve all of above troubles by introduce RELAX estimator


Reference:  "Backpropagation through the Void: Optimizing control variates for black-box gradient estimation"

Reference:  "REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models"
