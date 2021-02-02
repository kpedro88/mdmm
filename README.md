# mdmm

`mdmm` implements the Modified Differential Multiplier Method for PyTorch. It was proposed in Platt and Barr (1988), "[Constrained Differential Optimization](https://papers.nips.cc/paper/1987/file/a87ff679a2f3e71d9181a67b7542122c-Paper.pdf)". The MDMM minimizes a primary loss function subject to equality, inequality, and bound constraints on arbitrarily many secondary functions. It can be used for non-convex problems and problems with stochastic loss functions. It requires only one evaluation of the Lagrangian and its gradient per iteration, the same complexity as SGD.
