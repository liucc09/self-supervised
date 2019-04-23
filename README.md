# Self-supervised
The self supervised algorithm to train a neural network. The algorithm can take advantage of structured unlabeled samples to improve the performance of the network.

# Structure of the repository
The first version of the proposed method is implemented by Matlab and the second version is implemented by Pytorch, so there are both Matlab and Jupyter notebook files in this repository.

- neural core : the core units used to construct our network using matlab
- neural models : network and data model classes
- data_transfer : apis used to divide dataset and use PCA to transfer images to vectors
- data : the pumpkin dataset used in the paper
- ss_net.m : a example to use the proposed method
- coaxial error detection : a application example
- epfl_car : experiment on the epfl car dataset

# Results on EPFL Car Dataset

|Methods|MeanAE|MedianAE|Labeled|Unlabeled|
|:-----|-----|-----|-----|-----|
|Our approach 1 |9.28|3.5|1079|0|
|Our approach 2 |12.02|3.65|123|1389|
|Our approach 3 |17.22|4.78|123|0|
|Fenzi et al. (2015) |13.6|3.3|1179|0|
|He et al. (2014) |15.8|6.2|1179|0|
|Yang et al. (2017)|20.30|3.36|1179|0|
|Fenzi et al. (2014) |23.28|N/A|1179|0|
|Hara et al. (2017) |23.81|N/A|1179|0|
|Zhang et al. (2013) |24.00|N/A|1179|0|
|Yang et al. (2014) |24.1|3.3|1179|0|
|Hara et al. (2014) |24.24|N/A|1179|0|
|Fenzi et al. (2013) |31.27|N/A|1179|0|
|Torki et al. (2011) |33.98|11.3|1179|0|
|Teney et al. (2014) |34.7|5.2|1179|0|
|Redondo et al. (2014)  |39.8|7|1179|0|
|Ozuysal et al. (2009)  |46.5|N/A|1179|0|

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
