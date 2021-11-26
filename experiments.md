# [Poincaré Embeddings for Learning Hierarchical Representations](https://papers.nips.cc/paper/7213-poincare-embeddings-for-learning-hierarchical-representations)

## Summary

### Motivation and Proposals
- **Representation learning of symbolic data**. Want to embed symbolic data $\mathcal{S}$ to some "dense" space such as $\mathbb{R}^d$ so that the embedding preserves "structural relation" in the symbolic data. (The structure relation of embeded data is usualy "given" by the inner product.)
- Limitation of the Euclidean case: $\mathcal{S} \to \mathbb{R}^d$
  - When the symbolic data is complex, **large** embedding dimension is required for embedding to preserve the information of the data.
  - Focusing on symbolic data with hierarchical relations such as "hypernomy-hyponomy" relations in WordNet, the authors propose to use hyperbolic space as the target embedding space.
- Authors use Poincaré disk model $\left(\mathbb{B}^d = \{ \mathbf{x} \in \mathbb{R}^d : |\mathbf{x}| < 1\}, g(\mathbf{x}) = (\frac{2}{1 - |\mathbf{x}|^2})^2 g_0\right)$, where $g_0$ is the Euclidean metric tensor for the target hyperbolic space.
- Pros/Cons of embedding to Poincaré model
  - **Pros**
    - Bounded (as a **set**).
    - The distance between two points increases exponentially as the points going to the boundary of the disk. Hence, it is suitable to embed a graph whose number of nodes can grows exponetially with respect to the depth.
  - **Cons**
    - Numerical instability
      - Metric tensor near the boundary goes infinity.

### Formulation
- Consider the undirected graph $G = (S, A)$ consists of the set of nodes $S$ and the symmetric adjacent matrix $A$.
- $\Theta$ is the embedding of nodes $S$ into the $d$-dimensional Poincaré disk $\mathcal{B}^d$, i.e., $\Theta \in \mathbb{B}^{d\times{|S|}}$.
- Want to minimize a (problem specific) loss function $\mathcal{L}(\Theta)$ with respect to the embedding $\Theta$.
- More specifically, we consider the following loss function:
$$\mathcal{L}(\Theta) = -\frac{1}{|S|} \sum_{u \in S}\sum_{v \in P(u)} \log\left(\frac{e^{-d(\mathbf{u}, \mathbf{v})}}{\sum_{v' \in N(u)} e^{-d(\mathbf{u}, \mathbf{v'})}}\right)$$
- where
  - For each $u \in S$, $\mathbf{u}$ is the embedding of $u$ in the Poincaré disk.
  - $P(u)$ is the set of nodes directly connected with $u$.
  - $N(u) = S - P(u)$. Note that $u$ is also contained in $N(u)$. (In my opinion, it would be better to exclude $u$ itself from $N(u)$ for the better optimization. **CHECK**)
  - $d$ is the distance in the target space.

### Riemannian Optimization
- The above formulation is an optimization problem on the Riemannian space $(\mathcal{B}^d, g)$, **not** on the Euclidean space.
- For a general Riemannian manifold $(M, g)$, gradient vector field of a function $f: M \to \mathbb{R}$ is defined as follows:
  - We would like to find the direction (or unit vector "field") in which $f$ increases most rapidly. Note that the notion of **direction** (and also **unit** vector) depends on the notion of metric.
  - Differentiating $f$, we have differential $df: TM \to \mathbb{R}$, defined by $df(v_p) = \frac{d}{dt}|_{t=0} f(\gamma(t))$ for an arbitrary $\gamma(t)$ with $\gamma(0) = p$ and $\gamma'(0) = v_p$. Because of the non-degeneracy of the metric tensor $g$, there exists a unique vector field $\nabla^g f$ such that $g(\nabla^g f, X) = df(X)$ for any vector field $X$. This $\nabla^g f$ is called the **Riemannian gradient** of $f$. Note that, if we fix a local chart, than $\nabla^g f = g^{-1} \nabla f$ for the matrix $g$ and vector $\nabla^g f$ and $\nabla f$, where the matrix and vectors are expressed using the standard basis of the local chart.
- **Retraction**. To apply the 1st order optimization method (at a given point $p \in M$), choose a smooth map $R_p: T_p M \to M$ satisfying the following property:
  - $R_p (\mathbf{0}) = p$
  - $D_{\mathbf{0}}R_p(\mathbf{v}_p) = \mathbf{v}_p$ (=$D_{\mathbf{0}}\exp_p(\mathbf{v}_p)$ where $\exp$ is the exponetial map on $(M, g)$)
- **Riemannian Gradient Decent**. Update $p \leftarrow R_p(-\lambda\nabla^g f)$ for some step size $\lambda > 0$.
- In the Poincaré model case:
  - Since the metric tensor is the Euclidean metric with scaling factor $\left(\frac{2}{1 - |\mathbf{x}|^2}\right)^2$, we have $\nabla^g f = \left(\frac{1 - |\mathbf{x}|^2}{2}\right)^2 \nabla f$.
  - Use retraction $R_p : T_p \mathbb{B}^d \to \mathbb{B}^d$ defined by $R_p(\mathbf{v}) = p + \mathbf{v}$. Note that the operation $+$ may not be well-defined in general manifold case.






### Key modules in Code

#### `src/distance.py`: Distance Function on Poincaré Disk
##### `PoincareDistance` class
Note that the loss function can be computed from the distance terms (between every two embedded points).
This class is for defining distance function on Poincaré disk with *"new"* autograd.
The Poincaré distance between two points $\mathbf{u}, \mathbf{v} \in \mathbb{B}^d$ is as follows:
$$d(\mathbf{u}, \mathbf{v}) = \text{arccosh}\left(1 + 2 \frac{|\mathbf{u}-\mathbf{v}|^2}{(1 - |\mathbf{u}|^2)(1 - |\mathbf{v}|^2)}\right)$$

- The `forward` method computes distance (for Poincaré metric) between given two points.
$$\mathbb{B}^d \times \mathbb{B}^d \to \mathbb{R}$$
$$(\mathbf{u}, \mathbf{v}) \mapsto d(\mathbf{u}, \mathbf{v})$$
Note that the distance formula may give some numerically instable results if any point $\mathbf{u}$ or $\mathbf{v}$ is near the boundary of the Poincaré disk.
To remedy this, we clamp $|\mathbf{u}|^2$ and $|\mathbf{v}|^2$ into $[0, 1 - \epsilon)$ for some small $\epsilon > 0$.

- The `backward` method computes Euclidean gradient of the distance function. Denoting $z = d(\mathbf{u}, \mathbf{v})$ (and let $f$ be a given loss function),
$$\frac{df}{dz} \mapsto \left( \frac{df}{d\mathbf{u}}, \frac{df}{d\mathbf{v}} \right)$$

**Remark.**
The ordinary *autograd* of PyTorch also computes the Euclidean gradient.
However, the *autograd* and the above *backward* implementation might gives different values for the points near the boundary of the Poincaré disk.
This is because *autograd* of clamped value vanishes.


#### `src/optimizer.py`: Riemannian Gradient Decent Optimizer
##### `PoincareSGD` class
This class is for updating embedding vectors using Riemannian gradient decent algorithm on Poincaré disk.
Assuming that we have Euclidean gradient $\nabla \mathcal{L}$ from ordinary backpropagation (by calling `backward` method),
this optimizer does:
1. Compute Riemannian gradient $\nabla^g \mathcal{L}$ on the Poincaré disk from the Euclidean gradient $\nabla \mathcal{L}$.
1. Compute retraction $\mathbf{R}(-\lambda\nabla^g \mathcal{L})$.
1. Project the retraction point $\mathbf{x}$ to $\frac{\mathbf{x}}{|\mathbf{x}| + \epsilon}$ if $|\mathbf{x}| >= 1$ (outside of the Poincaré disk).
1. Update the embedding point to the above retracted (and projected) point.





## Experiments
Authors tested several symbolic data with hierarchical structure.
I only experimented with the subtree of "mammal" in the "hypernomy-hyponomy" tree of nouns in WordNet.

**Evaluation of the embedding**.
Author measured the quality of the embedding with two tasks: *reconstruction* and *link prediction*.
I only measured Mean Average Precision of the reconstruction.

### Vanila
- Hyperparams in the paper:
  - number of negative samples from $N(u)$: 10
  - burn-in epochs: 10
  - The ratio (burn-in learning rate) / (learning rate) = 0.1

### Reproduction Results
- The performance reported from the paper (for mammal subtree):
  - Setting: embedding dimension $d = 5$
  - MAP: 0.927
- The reproduced performance for $d = 5$:
  - MAP: 0.7128

### Observations
- Chosen paramters after several simple experiments
  - learning rate: 0.001
  - Use more than 20 negative samples: Using 50 negative samples gives the best score.
- Training process is highly unstable.
  - "U"-shape of training loss
  - Have to check numerical instability (maybe the train loss increases as the embedded vectors going closer to the boundary of the Poincaré disk ?)

### Discussion
- (Possible) Defects:
  - Negative Sampling
    - When the number of negative nodes $N(u)$ is smaller than the negative samples, sampled from $N(u)$ with "replace".
    - $u$ is included in the pool $N(u)$.
