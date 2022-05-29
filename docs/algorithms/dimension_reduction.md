Dimension reduction
===================

Single-cell ATAC-seq (scATAC-seq) produces large and highly sparse cell by feature count matrix.
Working directly with such a large matrix is very inconvinent and computational intensive.
Therefore typically, we need to reduce the dimensionality of the count matrix before
any downstream analysis. Most of the counts in this matrix are very small. For example,
~50% of the counts are 1 in deeply sequenced scATAC-seq data. As a result, 
many methods treat the count matrix as a binary matrix.

Different from most existing approaches, the dimension reduction method used in
SnapATAC2 is a pairwise-similarity based method, which requires defining and computing 
similarity between each pair of cells in the data.
This method was first proposed in [^Fang_2021], the version 1 of SnapATAC, and was called "diffusion map".
In SnapATAC2, we reformulate this approach as spectral embedding, *a.k.a.*, Laplacian eigenmaps.

Spectral embedding
------------------

Start with $n \times p$ cell by feature count matrix $M$, we first compute the
$n \times n$ pairwise similarity matrix $S$ such that $S_{ij} = \delta(M_{i*}, M_{j*})$,
where $\delta: \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}$ is the
function defines the similarity between any two cells. Typical choices of $\delta$
include the jaccard index and the cosine similarity.

We then compute the normalized graph Laplacian
$L = I - D^{-1/2} S D^{-1/2}$,
where $I$ is the identity matrix and $D$ is a diagonal matrix such that
$D_{ii} = \sum_k S_{ik}$.

The eigenvectors correspond to the k+1-smallest eigenvalues of $L$ are selected as
the lower dimensional embedding.

Nyström method
--------------

For samples with large numbers of cells, computing the full similarity matrix is
slow and requires a large amount of memory.
To address this limitation and increase the scalability of spectral embedding,
we used the Nystrom method to perform a low-rank approximation of the full
similarity matrix.

We will be focusing on generating an approximation $\tilde{S}$ of $S$ based on
a sample of $l ≪ n$ of its columns.

Suppose
$S = \begin{bmatrix}
A & B\\
B^T & C
\end{bmatrix}$
and columns $\begin{bmatrix} A \\ B^T \end{bmatrix}$ are our samples.
We first perform eigendecomposition on $A = U \Lambda U^T$.
The nystrom method approximates the eigenvectors of matrix $S$ by
$\tilde{U} = \begin{bmatrix}
U \\
B^T U \Lambda^{-1}
\end{bmatrix}$.

We can then compute $\tilde{S}$:

$$
\begin{align*}
\tilde{S} &= \tilde{U} \Lambda \tilde{U}^T \\
&= \begin{bmatrix}
U \\
B^T U \Lambda^{-1}
\end{bmatrix}\Lambda\begin{bmatrix}U^T & \Lambda^{-1}U^TB \end{bmatrix} \\
&= \begin{bmatrix}
U\Lambda U^T & U \Lambda \Lambda^{-1} U^T B \\
B^T U \Lambda^{-1} \Lambda U^T & B^T U \Lambda^{-1} \Lambda \Lambda^{-1} U^T B
\end{bmatrix} \\
&= \begin{bmatrix}
A & B \\
B^T & B^T U \Lambda^{-1} U^T B\end{bmatrix}
\end{align*}
$$

In practice, $\tilde{S}$ does not need to be computed.
Instead, it is used implicitly to estimate the degree normalization vector:

$$
\tilde{d} = \tilde{S}\mathbf{1} = \begin{bmatrix}A\mathbf{1} + B\mathbf{1} \\ B^T \mathbf{1} + B^T A^{-1} B\mathbf{1}
\end{bmatrix}
$$

[^Fang_2021]: Fang, R., Preissl, S., Li, Y. et al. Comprehensive analysis of single cell ATAC-seq data with SnapATAC. Nat Commun 12, 1337 (2021). https://doi.org/10.1038/s41467-021-21583-9