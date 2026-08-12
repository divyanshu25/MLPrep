# Foundations: Linear Algebra, Calculus, Probability - from first principles

The goal of this module: you should be able to *re-derive* everything here on
a whiteboard, not just recite it. Every ML system - a linear model, a
transformer, a diffusion model - is built out of exactly three ingredients:

1. **Linear algebra**: how we *represent* data and transform it (matrices).
2. **Calculus**: how we *improve* the transformation (gradients).
3. **Probability**: how we *justify* the loss function we're improving
   (likelihood).

If you deeply understand these three, "backprop through a transformer" stops
being magic and becomes bookkeeping.

Read in this order:

1. [Linear algebra for ML](01_linear_algebra.md) - vectors as data, the dot
   product as similarity and projection, matrix multiplication two ways,
   eigenvectors and the eigendecomposition, and the SVD with low-rank
   compression.
2. [Calculus for ML](02_calculus.md) - the derivative as local sensitivity,
   the chain rule as the whole of backprop, gradients of vector expressions,
   and Jacobians.
3. [Probability & statistics for ML](03_probability.md) - expectation and
   variance, Bayes rule, maximum likelihood (why cross-entropy is the loss),
   bias and variance of estimators, and the central limit theorem.
4. [Interview questions & exit checklist](04_interview_questions.md) - model
   answers to fifteen standard questions, and the checklist to clear before
   moving to the next module.

Then run the two notebooks, `gradients_from_scratch.ipynb` and
`probability_demos.ipynb`, and finish with [exercises.md](exercises.md).
