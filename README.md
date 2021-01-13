# Compositionally Warped Gaussian Processes
This package is dedicated to realizing method used in this [paper](https://arxiv.org/abs/1906.09665).

## Tutorial

Visit [here](./examples/cwgp_beta.ipynb)

## Transformations

### Sinh-Arcsinh (sa)

`from cwgp.transformations import sa`

### Arcsinh (asinh)

`from cwgp.transformations import asinh`

### Box-Cox (box_cox)

`from cwgp.transformations import box_cox`

### Sinh-Arcsinh and Affine (SAL)

`from cwgp.transformations import sal`

## Kernels

### Ornstein-Uhlenbeck Kernel

`from cwgp.kernel import OU`

### Radial Basis function Kernel

`from cwgp.kernel import RBF`
