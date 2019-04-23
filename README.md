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

# Reference
If you found these codes useful for your research, please consider citing - https://ieeexplore.ieee.org/abstract/document/8664135
```
@article{liu2019self,
  title={Self-supervised learning for specified latent representation},
  author={Liu, ChiCheng and Song, Libin and Zhang, Jiwen and Chen, Ken and Xu, Jing},
  journal={IEEE Transactions on Fuzzy Systems},
  year={2019},
  publisher={IEEE}
}
```
