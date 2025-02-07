---
title: "Quantum Principal Component Analysis and Self-Tomography"
datePublished: Fri Feb 07 2025 21:43:15 GMT+0000 (Coordinated Universal Time)
cuid: cm6vamd3o00030al4bk8bhfze
slug: qpca
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/ISHD1ovpJ-k/upload/8a0043530ae3f4bcb5db825afea6fc3a.jpeg
tags: qml, pca, quantum-machine-learning-qml, qpca

---

High-dimensional data presents significant analytical challenges, for example some algorithms suffer the curse of dimensionality (i.e. as the number of dimensions increases, the volume of the data space grows exponentially, making the computation expensive or even unfeasible), while the presence of many features might result in models becoming overly complex and learning noise instead of true patterns.

One of the most well known and powerful techniques to address high-dimensionality is known as Principal Component Analysis (PCA), a statistical technique used to simplify complex datasets by reducing their dimensionality while preserving the most important information. This blog post discusses PCA, focusing on the selection of the principal components, and then introduce a quantum circuit performing quantum Principal Component Analysis, a quantum algorithms providing an exponential speedup over PCA.

## Principal Component Analysis

Principal Component Analysis (PCA), also known as the Karhunen-Loève transformation, the Hotelling transformation or the method of empirical orthogonal functions, aims to project \\(p\\)\-dimensional vectors to the so-called principal components, i.e. \\(q\\)\-dimensional vectors, where \\(q<p\\).

There are several equivalent ways of deriving the principal components mathematically and the following section shows that finding the projectors maximizing the variance is equivalent to minimizing the means squared distance between the original vectors and their projections on to the principal components.

### Mathematics of principal components

Let a centered matrix \\(X \in C^{n\times p}\\) and let \\(\{x_i\}_{i=1}^n\\) be \\(p\\)\-dimensional vectors (i.e. they represent the columns of \\(X\\)). The projection of the \\(x_i\\) on to a line (\\(w\\)) through the origin (for simplicity) is:

\\[(x_i ^\dagger w) w\\]

It’s relevant to note that the mean of the projections is zero being the vectors \\(x_i\\) centered:

\\[\frac 1n \sum_i (x_i ^\dagger w) w=\frac 1n \sum_i (x_i)^\dagger w w\\]

Being a projection, the projected vectors are (in general) different from the original vectors, which means there’s some error. Such error is defined as:

\\[\begin{aligned}\left\|{x_i}-\left({w} ^\dagger {x_i}\right) {w}\right\|^2= & \left({x_i}-\left({w} ^\dagger {x_i}\right) {w}\right) ^\dagger\left({x_i}-\left({w} ^\dagger {x_i}\right) {w}\right) \\ = & {x_i} ^\dagger {x_i}-{x_i} ^\dagger\left({w} ^\dagger {x_i}\right) {w} \\ & -\left({w} ^\dagger {x_i}\right) {w} ^\dagger {x_i}+\left({w} ^\dagger {x_i}\right) {w} ^\dagger\left({w} ^\dagger {x_i}\right) {w} \\ = & \left\|{x_i}\right\|^2-2\left({w} ^\dagger {x_i}\right)^2+\left({w} ^\dagger {x_i}\right)^2 {w} ^\dagger {w} \\ = & {x_i} ^\dagger {x_i}-\left({w} ^\dagger {x_i}\right)^2\end{aligned}\\]

Since the vector \\(w\\) is defined as normal, the mean squared error (MSE) is equivalent to:

\\[\text {MSE}=\frac 1n \sum_{i=1}^n {x_i} ^\dagger {x_i}-\left({w} ^\dagger {x_i}\right)^2\\]

Considering that the first inner product doesn’t involve \\(w\\) and is therefore a constant, minimizing the MSE is then equivalent to:

\\[\text{max}_w \frac 1n\sum_{i=1}^n (w^\dagger x_i)^2\\]

Since the mean of a square is always equal to the square of the mean plus the variance, the function to be maximized is equivalent to:

\\[\text{max}_w \frac 1n\sum_{i=1}^n (w^\dagger x_i)^2 = \text{max}_w \left(\frac 1n \sum_{i=1}^n x_i ^\dagger w\right)^2 + \text{var}(w^\dagger x_i)\\]

However, since that the mean of the projections is zero (see the above), minimizing the residual sum of squares turns out to be equivalent to maximizing the variance of the projections.

This is true also if we don’t want to project on to just one vector, but on to multiple principal components.

Accordingly then, the variance \\(\sigma\left({{w}}^2\right) \\) is defined (in matrix form) as:

\\[\begin{aligned} \sigma^2 \left({{w}}\right) & =\frac{1}{n} \sum_i\left({x_i} ^\dagger {w}\right)^2 \\ & =\frac{1}{n}({x w})^\dagger({x w}) \\ & =\frac{1}{n} {w}^\dagger {x}^\dagger {x w} \\ & ={w}^\dagger \frac{{x}^\dagger {x}}{n} {w} \\ & ={w}^\dagger {V w}\end{aligned}\\]

where \\(V\\) is the covariance matrix of \\(x\\).

Therefore the constrained maximization problem is:

\\[\text{max}_w \space\sigma^2(w) \space \text{s.t.} \space w^\dagger w =1\\]

Using the Lagrange multiplier \\(\gamma\\), the objective function becomes:

\\[L(\gamma, w) = w^\dagger V w-\gamma(w^\dagger w-1)\\]

The first order conditions are:

\\[\begin{align} & \frac {\partial L}{\partial w}\ = 2Vw-2\gamma w\\ & \frac{\partial L}{\partial \gamma}\ = w^\dagger w-1\end{align}\\]

Setting the derivatives to zero at optimum the system becomes:

\\[\begin{align} & Vw=\gamma w\\ &w^\dagger w=1\end{align}\\]

and from the top equation is clear that the \\(w\\) maximizing the variance are the orthonormal eigenvectors of the covariance matrix associated with the largest \\(q\\) eigenvalues \\(\gamma\\).

It’s clear that if the data are approximately \\(q\\)\-dimensional (i.e. \\(p-q\\) eigenvalues are close to 0), the residual will be small and the \\(R^2\\) (the fraction of the original variance of the dependent variable kept by the fitted values), computed as:

\\[R^2 = \frac{\sum_{i=1}^q \gamma_i}{\sum_{i=1}^p \gamma_i}\\]

will be close to 1.

### Complexity analysis of PCA

Assuming \\(X\in C^{n\times p}\\), the cost of PCA is:

* computing \\(V\\)is \\(\mathcal O(n\times p^2)\\)
    
* computing the eigenvalues and eigenvectors requires \\(\mathcal O(p^3)\\)
    

Hence the overall complexity is \\(\mathcal O(n\times p^2 + p^3) \approx \mathcal O(p^3)\\).

## Quantum Principal Component Analysis

The idea behind Quantum Principal Component Analysis (qPCA) is to use quantum subroutines to perform PCA faster. In particular, the idea is to use Quantum Phase Estimation (QPE) to get information about the eigenvalues and the eigenvectors of a density matrix representing the covariance matrix. The next section is therefore about introducing QPE, while the following sections discuss qPCA.

### Quantum Phase Estimation

Let \\(U\\) a unitary operator and let \\(\ket{u_k}\\) and \\(e^{i\lambda_k}\\) be the \\(k\\)\-th eigenvector and eigenvalue of \\(U\\). Assume also a generic state \\(\ket \psi\\), which can always be defined as:

\\[\ket \psi= \sum_{k=1}^nc_k\ket {u_k}\\]

The goal of the QPE is to perform the following transformation:

\\[\text{QPE}:\ket{0}^{\otimes n}\ket{\psi} \rightarrow \sum_{k=1}^nc_k\ket{\lambda_k }\ket{u_k}\\]

where \\(\ket {\lambda_k}\\) is the quantum state \\(\ket {j_1\dots j_n}\\) corresponding to \\(n\\) digits representation of the binary fractional representing the eigenvalue phase.

The circuit of the algorithm is the following:

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1738960092600/39527e43-e9d1-4870-92da-24ec82088e82.png align="center")

First, each of the nn qubits initialized to \\(\ket 0\\) is subjected to the Hadamard gate and control unitary operations acting on \\(\ket u\\), which perform the following transformation:

\\[\ket 0^{\otimes n}\ket u \rightarrow \bigotimes_{k=1}^n \left(\frac{\ket 0 +e^{i2\pi0.j_1\dots j_k}\ket {1}}{\sqrt 2} \right) \otimes\ket u\\]

In other words, binary decimal representation of the phase of the eigenvalue is stored in the phase of each auxiliary qubit, with digits shift one position repeatedly.

The states of the \\(n\\) auxiliary qubits have exactly the same form as the expression for the result of the quantum Fourier transform, therefore applying the inverse quantum Fourier transform and measuring the \\(n\\)ancilla bit results in \\(0.j_1\dots j_n\\) which the \\(\lambda\\).

### Density matrix exponentiation

One can then imagine that if we’re able to encode the covariance matrix as a quantum gate, we can use QPE to obtain information regarding the eigenvalues and the eigenvectors. That’s indeed true, however the most important property of quantum gates is unitarity, i.e. for any quantum gate \\(G\\):

\\[G^\dagger= G^{-1}\\]

This however is not generally true for covariance matrices, however, one can make a covariance matrix unitary by using exponentiation. Assume the covariance matrix has been encoded in a density matrix \\(\rho\\) and assume one is presented with \\(n\\) copies of \\(\rho\\). The density matrix exponential:

\\[e^{-i\rho t}\\]

is unitary.

One method to perform such exponentiation up to \\(n\\)\-th order in \\(t\\) is to repeat the following:

\\[\text{Tr}_p \left[e^{-iS\Delta t} \otimes \sigma \otimes e^{iS\Delta t}\right] = \sigma -i\Delta t[\rho, \sigma]+\mathcal O(\Delta t^2)\\]

where \\(\sigma\\) is any density matrix, \\(S\\) is the swap operator and \\([A, B] = A-B\\) and \\(\text{Tr}_p\\) is the partial trace on the first system. It’s worth to note that since \\(S\\) is a sparse matrix, the exponentiation of \\(S\\) can be computed efficiently. Applying the above formula \\(n\\) times leads to:

\\[e^{-i\rho n\Delta t} \otimes \sigma \otimes e^{i\rho n\Delta t}\\]

which, couple with the quantum matrix inversion technique of ([Harrow, Hassidim, Lloyd, 2009, “Quantum algorithm for linear systems of equations“](https://arxiv.org/pdf/0811.3171)) allows to efficiently construct the exponential of \\(\rho\\).

So, assuming a non-sparse positive \\(X\\) whose trace is 1, to construct:

\\[e^{-iXt}\\]

requires to factor \\(X\\) as:

\\[X=A^\dagger A=\sum_i |a_i|\ket{\hat a_i}\ket {e_i}\\]

where \\(A=\sum_i |a_i|\ket{\hat a_i}\ket {e_i}\\) \\(\hat a_i\\) is the version of \\(a_i\\) normalized to 1, the columns of \\(A\\) and \\(\ket {e_i}\\) is an orthonormal basis. Assuming a qRAM ([Giovannetti, Lloyd, Maccone, 2008, “Quantum random access memory“](https://arxiv.org/pdf/0708.1879)) that performs the following:

\\[\ket i \ket 0 \ket 0 \rightarrow \ket i\ket {\hat a_i} \ket {|a_i|}\\]

one can easily construct the state \\(\ket \psi = \sum_i |a_i|\ket {e_i}\ket {\hat a_i}\\), whose density matrix is:

\\[(\sum_i |a_i|\ket {e_i}\ket {\hat a_i})\otimes (\sum_i |a_i|\bra{e_i}\bra{\hat a_i}) = X\\]

So by using \\(n=\mathcal O(t^2\epsilon^{-1})\\) copies of \\(X\\) one can implement \\(e^{-iXt}\\) with accuracy \\(\epsilon\\) in \\(\mathcal O(n\log d)\\) time.

### Obtaining the Principal Components using self-tomography

Once the exponentiation of the covariance matrix is performed, one can use QPE to find the eigenvectors and eigenvalues of the density matrix using conditional application of:

\\[e^{-iXt}\\]

for varying times \\(t\\), using \\(\ket \psi\\) as the initial state, resulting in the following state:

\\[\sum_ir_i\ket{\chi_i}\bra {\chi_i}\ket{\tilde r_i}\bra{\tilde r_i}\\]

where \\(\ket {\chi_i}\\) are the eigenvalues of \\(X\\) and \\(\ket {\tilde r_i}\\) are the corresponding eigenvalues.

The extraction of \\(i\\)\-th eigenvalues are then determined by measuring the expectation value of the eigenvector with eigenvalue \\(r_i\\):

\\[\bra{\chi_i}M \ket{\chi_i}\\]

This process, called quantum self-tomography, reveals eigenvalues and eigenvectors in time \\(\mathcal O(R\log d)\\), where \\(R\\) is the rank of \\(X\\), resulting in an exponential speedup over classical PCA.

---

And that's it for this article. Thanks for reading.

For any question or suggestion related to what I covered in this article, please add it as a comment. For special needs, you can contact me [**here.**](http://amm.zanotp.com/contact)

## Sources:

* [Lloyd, Mohseni, Rebentrost, “Quantum principal component analysis”, 2014](https://arxiv.org/pdf/1307.0401)
    
* [He, Li, Liu, Wang, “A Low Complexity Quantum Principal Component Analysis Algorithm”, 2021](https://arxiv.org/pdf/2010.00831)
    
* [Nghiem, “New Quantum Algorithm for Principal Component Analysis”, 2025](https://arxiv.org/pdf/2501.07891v1)
    
* [Harrow, Hassidim, Lloyd, 2009, “Quantum algorithm for linear systems of equations“](https://arxiv.org/pdf/0811.3171)
    
* [Giovannetti, Lloyd, Maccone, 2008, “Quantum random access memory“](https://arxiv.org/pdf/0708.1879)