# ScPace: Timestamp Calibration for time-series ScRNA-seq expression data

We present ScPace, a novel approach for timestamp calibration in time-series single-cell RNA sequencing (scRNA-seq) data. This method tackles the significant issue of noisy timestamps in time-series ScRNA-seq data, which can undermine the accuracy of timestamp automatic annotation (TAA) and interfere with downstream analyses, such as supervised pseudotime analysis. ScPace leverages a latent variable indicator within a support vector machine (SVM) framework to efficiently identify and correct mislabeled samples, improving the robustness and reliability of results across a range of time-series ScRNA-seq datasets.

# Input

- `data`: The input time-series ScRNA-seq expression matrix.
- `labels`: Time-labels for each cells(Numpy Series)
- `C`: The regularization parameter for SVM
- `num_iteration`: Numbers of iterations to perform ScPace
- `p`: The C-Growing Parameter
- `lam`: Threshold for updating the latent variables
- `methods`: reclassify/deletion

## Paper
[Timestamp calibration for time-series single cell RNA-seq expression data](https://arxiv.org/abs/2412.03027)
