**The Self-Pruning Neural Network**

**Overview**

In modern
machine learning systems, deploying large neural networks is often constrained
by memory and computational resources. A common solution is  **model pruning** ,
where unnecessary weights are removed after training.

In this
project, we extend this idea by designing a **self-pruning neural network**
that learns which connections to remove  *during training* , rather than as
a post-processing step. This is achieved by introducing learnable gate
parameters for each weight and enforcing sparsity through regularization.

**Methodology**

**1.
Prunable Layers**

We
implemented custom layers (PrunableLinear and PrunableConv2d) where each weight
is associated with a learnable parameter called gate_score.

During the
forward pass:

* Gate values are computed as:

**g =
sigmoid(gate_score)**

* The effective weight becomes:

 **W_pruned
= W ** **⊙****
g**

If a gate
approaches zero, the corresponding weight is effectively removed, allowing the
network to dynamically suppress unimportant connections.

**2. Sparsity-Inducing Loss**

To
encourage pruning, we modify the loss function:

**Total
Loss = CrossEntropyLoss + λ × SparsityLoss**

Where:

* **SparsityLoss = sum of all gate
  values**

Since gate
values lie in the range (0, 1), applying an **L1 penalty** encourages many
of them to shrink toward zero.

**3. Why L1 Encourages Sparsity**

The L1
norm applies a constant penalty to all values, regardless of magnitude. This
makes it more effective than L2 regularization in driving parameters exactly to
zero.

In this
context:

* Small gate values are pushed
  aggressively toward zero
* Important connections survive
  because removing them would significantly increase classification loss

This
creates a natural selection mechanism where only the most useful weights remain
active.

**Experimental Setup**

* Dataset: CIFAR-10
* Model: Custom CNN with fully
  prunable convolutional and linear layers
* Optimizer: Adam
* Training: 20 epochs
* λ values tested: {1e-5, 5e-5,
  1e-4}

**Results**

|  **Lambda**  |  **Test
   Accuracy (%)**  |  **Sparsity

| (%)** |       |       |
| ----- | ----- | ----- |
| 1e-5  | 84.48 | 61.65 |
| 5e-5  | 84.17 | 58.55 |
| 1e-4  | 83.95 | 71.84 |

**Analysis**

* Increasing λ generally leads
  to higher sparsity, demonstrating effective control over the pruning
  strength
* The network retains high
  accuracy (~84%) even with up to  **~72% of weights pruned** , indicating
  significant redundancy in the original model
* Pruning does not occur
  gradually; instead, a **phase transition** is observed during training,
  where sparsity increases sharply after a certain point
* This transition typically
  occurs after the model has already learned useful features (around epoch
  13–14), suggesting that the network prioritizes learning before optimizing
  for sparsity
* Moderate pruning acts as a
  form of  **regularization** , helping maintain strong generalization
  performance

**Gate Value Distribution**

![]()

The
distribution shows:

* A large spike near 0 →
  indicating a high number of pruned weights
* A smooth tail of non-zero
  values → reflecting varying degrees of importance among remaining
  connections

This
indicates that the model performs  **soft pruning** , assigning continuous
importance scores rather than enforcing strict binary decisions.

**Conclusion**

This
project demonstrates that:

* Neural networks can learn
  their own optimal sparse structure during training
* L1 regularization on
  sigmoid-based gates effectively induces sparsity
* Significant model compression
  (~70%) can be achieved with minimal loss in accuracy
* Pruning emerges naturally as
  part of the optimization process rather than requiring a separate
  post-training step

This
approach provides a principled method for building efficient neural networks
suitable for deployment under resource constraints.

**Future Work**

* Structured pruning
  (channel/filter-level pruning)
* Hard gating mechanisms (binary
  masks during training)
* Sparse tensor optimization for
  faster inference
* C++ inference engine for
  optimized deployment
