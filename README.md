# self-supervised
The self supervised algorithm to train a neural network. The algorithm can take advantage of structured unlabeled samples to improve the performance of the network.

# structure of the repository
The first version of the proposed method is implemented by Matlab and the second version is implemented by Pytorch, so there are both Matlab and Jupyter notebook files in this repository.

- neural core : the core units used to construct our network using matlab
- neural models : network and data model classes
- data_transfer : apis used to divide dataset and use PCA to transfer images to vectors
- data : the pumpkin dataset used in the paper
- ss_net.m : a example to use the proposed method
- coaxial error detection : a application example
- epfl_car : experiment on the epfl car dataset
