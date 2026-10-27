# Foundations: Linear Algebra, Calculus, Probability - from first principles

The goal of this module: you should be able to *re-derive* everything here on
a whiteboard, not just recite it. Every ML system - a linear model, a
transformer, a diffusion model - is built out of exactly three ingredients:

1. **Linear algebra**: how we *represent* data and transform it (matrices).
2. **Calculus**: how we *improve* the transformation (gradients).
3. **Probability**: how we *justify* the loss function we're improving
   (likelihood).

With these three in hand, "backprop through a transformer" is a sequence of
chain-rule steps you can carry out by hand.

Read in this order:

1. [Linear algebra for ML](02_linear_algebra.md) - vectors as data, the dot
   product as similarity and projection, matrix multiplication two ways,
   eigenvectors and the eigendecomposition, the SVD with low-rank
   compression, and norms.
2. [Calculus for ML](03_calculus.md) - the derivative as local sensitivity,
   the chain rule as the whole of backprop, gradients of vector expressions,
   and Jacobians.
3. [Probability & statistics for ML](04_probability.md) - expectation and
   variance, the Bernoulli and Gaussian distributions, Bayes rule, maximum
   likelihood (where cross-entropy and MSE come from), information theory
   (entropy, cross-entropy, KL), error bars and the central limit theorem,
   bias and variance of estimators, and hypothesis tests.
4. [Interview questions & exit checklist](05_interview_questions.md) - model
   answers to fifteen standard questions, and the checklist to clear before
   moving to the next module.

Then run the two notebooks, `gradients_from_scratch.ipynb` and
`probability_demos.ipynb`, and finish with [exercises.md](exercises.md).
