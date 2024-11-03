# TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning (ICML2024) [![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/abs/2405.03140)

Paper link (preprint): [https://arxiv.org/abs/2405.03140]

## News :fire:
- **May 9, 2024:** Congratulations ! Paper has been accepted by ICML 2024 !
- **Oct 5, 2024:** We updated the tool to efficiently summarize experimental results ```result_extractor.py```. see below.
- **Nov 2,2024:** Check some pretrained weights [Link](https://drive.google.com/file/d/1st_Nbjl9KzmFXeeLV4x4Ch9PZY7FMEmc/view?usp=sharing). 
<img align="right" width="50%" height="100%" src="https://github.com/xiwenc1/TimeMIL/blob/main/Figs/intro2_1.jpg">

> **Abstract.** Deep neural networks, including transformers and convolutional neural networks, have significantly improved multivariate time series classification (MTSC). However, these methods often rely on supervised learning, which does not fully account for the sparsity and locality of patterns in time series data (e.g., diseases-related anomalous points in ECG). To address this challenge, we formally reformulate MTSC as a weakly supervised problem, introducing a novel multiple-instance learning (MIL) framework for better localization of patterns of interest and modeling time dependencies within time series. Our novel approach, TimeMIL, formulates the temporal correlation and ordering within a time-aware MIL pooling, leveraging a tokenized transformer with a specialized learnable wavelet positional token. The proposed method surpassed 26 recent state-of-the-art methods, underscoring the effectiveness of the weakly supervised TimeMIL in MTSC. 




> **Method.** The proposed framework of TimeMIL for time series classification with enhanced interpretability: (i) a feature extractor to obtain instance-level feature embeddings, (ii) a MIL pooling to aggregate instance embeddings to a bag-level feature, embedding, and (iii) a bag-level classifier to map bag-level feature to a label prediction. Each time point is treated as an instance and the time series as a bag. Time ordering information and instance correlation are captured by taking the mutual benefit of WPE and MHSA in our TimeMIL pooling (highlighted in green).
<img src="https://github.com/xiwenc1/TimeMIL/blob/main/Figs/network_v2.jpg">


## Dependencies
```
aeon                      0.5.0
numpy                     1.23.1
torch                     1.13.1+cu117
torchvision               0.14.1+cu117
python                    3.8.18
```



## Usage



## Hyperparamter strategy for custom datasets
```dropout_patch```: {0.1,0.5} the ratio of windows are randomly masked in each iteration (see appendix F.3).
```epoch_des```: {0,10,20} the number of epoches for warm-up (see appendix F.4).
```batchsize```: {8,16,32,64,128}.

## Summarize results
Modify line 11 and 14 in ``` result_extractor.py```

```
path = Your_output_folder_Path  #'./{args.save_dir}/InceptBackbone/'
outpath = Path_to_output_the_summarized_results    #'savemodel2_csv/' 
```
Then, run the code:
```python result_extractor.py```

## Pretrained weights

We show some examples of weights with different (good) Hyperparamter here:
[Link](https://drive.google.com/file/d/1st_Nbjl9KzmFXeeLV4x4Ch9PZY7FMEmc/view?usp=sharing)
 

## Citation
If you find our work is useful in your research, please consider raising a star  :star:  and citing:

```
@article{chen2024timemil,
  title={TimeMIL: Advancing Multivariate Time Series Classification via a Time-aware Multiple Instance Learning},
  author={Chen, Xiwen and Qiu, Peijie and Zhu, Wenhui and Li, Huayu and Wang, Hao and Sotiras, Aristeidis and Wang, Yalin and Razi, Abolfazl},
  journal={arXiv preprint arXiv:2405.03140},
  year={2024}
}
```



## Thanks for the code provided by:
- Todynet:https://github.com/liuxz1011/TodyNet
- TapNet: https://github.com/kdd2019-tapnet/tapnet
- OS-CNN: https://github.com/Wensi-Tang/OS-CNN
- Nystromformer: https://github.com/mlpen/Nystromformer
- Group 2 Experiment from: https://github.com/thuml/Time-Series-Library/tree/main
