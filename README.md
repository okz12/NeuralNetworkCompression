# NeuralNetworkCompression

This repository investigates two methods of compressing neural networks:

1. Knowledge Distillation

2. Soft-weight Sharing with Pruning

## Knowledge Distillation

Knowledge distillation uses a larger teacher model to train and smaller student model.

### Training

The output logits from the larger model are used to train the smaller model instead of one-hot encoded targets.

![KD NN](figs/gifs/KD_NN.gif?raw=true "KD NN")

### Student Targets

Dark knowledge refers to the information hidden in the tail end of the probability distribution. Teacher logits (<img src="https://render.githubusercontent.com/render/math?math=$z$&mode=inline">) can be smoothed by a temperature parameter (<img src="https://render.githubusercontent.com/render/math?math=$T$&mode=inline">) to give a softer output distribution to be used as student targets (<img src="https://render.githubusercontent.com/render/math?math=$q$&mode=inline">). A smaller model finds it easier to match the softer output distribution, resulting in higher accuracy. A lower temperature results in a sharper information, a higher temperature pushes up the tail of the distribution but approaches a uniform distribution at very high temperature values.

![KD Dark Knowledge](figs/gifs/KD_Dark_Knowledge.gif?raw=true "KD Dark Knowledge")

### Results

I ran the experiments on a Custom MiniVGG, which uses structure and blocks of VGG but is smaller than VGG-11. The number of layers and parameters is given below. Using knowledge distillation results in an uplift of 1.7% in accuracy on the CIFAR10 dataset.

|                      | VGG-13    | Custom MiniVGG | Custom MiniVGG + VGG13 KD |
|----------------------|-----------|----------------|---------------------------|
| Number of Layers     | 36        | 21             | 21                        |
| Number of Parameters | 9,416,010 | 1,575,690      | 1,575,690                 |
| Accuracy             | 85.5%     | 79.7%          | 81.4%                     |

## Soft-weight Sharing

Soft-weight sharing relies on imposing priors on the weights of the neural network as penalties, reducing the size of the network through optimisation.

![SWS NN](figs/gifs/SWS_NN.gif?raw=true "SWS NN")

### Priors

Soft-weight sharing imposes a Gaussian Mixture Model (GMM) parameter on the network forcing it to sparsity:

<img src="https://render.githubusercontent.com/render/math?math=\text{Accuracy Loss} \times \text{Clustering Loss} \times \text{Trade-off Parameter}&mode=inline"> =
<img src="https://render.githubusercontent.com/render/math?math=-\frac{1}{N} \sum_{i=1}^{N}y_i \log (\hat{y_i}) \times \tau \times \sum_{i=1}^{N} \log \sum_{j=0}^{J} \pi_j \mathcal{N}(w_i | \mu_j, \sigma_j^2)&mode=inline">

Imposing the GMM prior is very similar to imposing a Gaussian prior for L2-regularisation. With L2-regularisation and Gaussian distribution penalty is added to the loss function, forcing the weights to cluster around 0.

L2-Prior Penalty term: <img src="https://render.githubusercontent.com/render/math?math=p(w) = \sum_{i=1} \mathcal{N}(w_i | 0, \sigma^2)&mode=inline">

![SWS Prior L2](figs/gifs/SWS_Priors_L2.gif?raw=true "SWS Prior L2")

The GMM loss also has a cluster fixed at 0 to enforce sparsity, but allows other free-clusters with variable means to allow the network to retain values further from 0. The cluster at 0 is also given a higher mixing proportion (<img src="https://render.githubusercontent.com/render/math?math=\pi_0&mode=inline">) to ensure most of the network weights remain in the 0-mean cluster.

GMM-Prior Penalty term: <img src="https://render.githubusercontent.com/render/math?math=\small{p(w) = \prod_{i=1}^{N} \sum_{j=0}^{J} \pi_j \mathcal{N}(w_i | \mu_j, \sigma_j^2)}&mode=inline">

![SWS Prior GMM](figs/gifs/SWS_Priors_GMM.gif?raw=true "SWS Prior GMM")

### Steps

Training for soft-weight sharing involves 3 steps.

#### Step 1: Training

The first step is to train the network normally (i.e. without any added penalties for clustering to the loss function).

![SWS SWS Clustering Step 1 Training](figs/gifs/SWS_Clustering_1_training.gif?raw=true "SWS Clustering Step 1 Training")

#### Step 2: Clustering

The second step is to add the clustering penalty to the loss function which transforms the network weight distribution to a GMM.

![SWS Clustering Step 2 Clustering](figs/gifs/SWS_Clustering_2_clustering.gif?raw=true "SWS Clustering Step 2 Clustering")

#### Step 3: Prune and Quantize

The final step is to prune and quantize each of the mixtures to their mixture means. The 0-mean mixture is effectively pruned and the remaining values are quantized and can be represented very compactly using codebook quantization.

![SWS Clustering Step 3 Prune and Quantize](figs/gifs/SWS_Clustering_3_prune_quantize.gif?raw=true "SWS Clustering Step 1 Prune and Quantize")

### Results

Below are the results for implementing soft-weight sharing on a LeNet-300-100 with MNIST dataset.

The joint plot

![Lenet Joint Plot](figs/lenet_jp.gif?raw=true "Lenet Joint Plot")

![Lenet SWS Weights](figs/lenet_sws_weights.gif?raw=true "Lenet SWS Weights")

|          | Original | SWS   |
|----------|----------|-------|
| Accuracy | 98.2%    | 96.9% |
| Sparsity | 0.0%     | 98.0% |



## Sources
- Hinton, G., Vinyals, O. and Dean, J., 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531.
- Ullrich, K., Meeds, E. and Welling, M., 2017. Soft weight-sharing for neural network compression. arXiv preprint arXiv:1702.04008.
- Nowlan, S.J. and Hinton, G.E., 1992. Simplifying neural networks by soft weight-sharing. Neural computation, 4(4), pp.473-493.
