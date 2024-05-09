# TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning (ICML2024)


Paper link (preprint): [https://arxiv.org/abs/2405.03140]

## Abstract

Deep neural networks, including transformers and convolutional neural networks, have significantly improved multivariate time series classification (MTSC). However, these methods often rely on supervised learning, which does not fully account for the sparsity and locality of patterns in time series data (e.g., diseases-related anomalous points in ECG). To address this challenge, we formally reformulate MTSC as a weakly supervised problem, introducing a novel multiple-instance learning (MIL) framework for better localization of patterns of interest and modeling time dependencies within time series. Our novel approach, TimeMIL, formulates the temporal correlation and ordering within a time-aware MIL pooling, leveraging a tokenized transformer with a specialized learnable wavelet positional token. The proposed method surpassed 26 recent state-of-the-art methods, underscoring the effectiveness of the weakly supervised TimeMIL in MTSC. 


##

<img src="https://github.com/xiwenc1/TimeMIL/blob/main/Figs/intro2_1.jpg" width="500">

**(a):** The decision boundary of fully supervised methods is determined by assigning a label to each time series. **(b):** TimeMIL makes decisions by discriminating positive and negative instances in time series, where each time point is an instance, and its label is typically not available in reality.
##

<img src="https://github.com/xiwenc1/TimeMIL/blob/main/Figs/network_v2.jpg" width="800">

The proposed framework of TimeMIL for time series classification with enhanced interpretability: (i) a feature extractor to obtain instance-level feature embeddings, (ii) a MIL pooling to aggregate instance embeddings to a bag-level feature, embedding, and (iii) a bag-level classifier to map bag-level feature to a label prediction. Each time point is treated as an instance and the time series as a bag. Time ordering information and instance correlation are captured by taking the mutual benefit of WPE and MHSA in our TimeMIL pooling (highlighted in green).



## Dependencies




## Usage





## Hyperparamters for custom datasets
```dropout_patch```: {0.1,0.5} the ratio of windows are randomly masked in each iteration (see appendix F.3).
```epoch_des```: {0,10,20} the number of epoches for warm-up (see appendix F.4).
```batchsize```: {8,16,32,64,128}.
