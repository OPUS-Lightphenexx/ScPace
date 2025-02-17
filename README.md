# ScPace: Timestamp Calibration for time-series ScRNA-seq expression data

We present ScPace, a novel approach for timestamp calibration in time-series single-cell RNA sequencing (scRNA-seq) data. This method tackles the significant issue of noisy timestamps in time-series ScRNA-seq data, which can undermine the accuracy of timestamp automatic annotation (TAA) and interfere with downstream analyses, such as supervised pseudotime analysis. ScPace leverages a latent variable indicator within a support vector machine (SVM) framework to efficiently identify and correct mislabeled samples, improving the robustness and reliability of results across a range of time-series ScRNA-seq datasets.

![graph_abstract](https://github.com/user-attachments/assets/bf96babc-f313-4e26-a036-66eff6f1c62c)
# Files
- `ScPace/ScPace`: Main code for ScPace.
- `ScPace/Hinge_Loss`: Computes Hinge Loss for each Samples
- `Simulated Datasets`: Simulated datasets using Splatter


# Input

- `data`: The input time-series ScRNA-seq expression matrix.
- `labels`: Time-labels for each cells(Numpy Ndarray)
- `C`: The regularization parameter for SVM
- `num_iteration`: Numbers of iterations to perform ScPace
- `p`: The C-Growing Parameter
- `lam`: Threshold for updating the latent variables
- `methods`: reclassify/deletion

## Paper
Arxiv: [Timestamp calibration for time-series single cell RNA-seq expression data](https://arxiv.org/abs/2412.03027)
Researchgate: [Timestamp calibration for time-series single cell RNA-seq expression data]([https://arxiv.org/abs/2412.03027](https://www.researchgate.net/publication/386426329_Timestamp_calibration_for_time-series_single_cell_RNA-seq_expression_data))
Supplementary files: [Supplementary](https://www.researchgate.net/publication/389051068_SupplementaryScPacedocx?_tp=eyJjb250ZXh0Ijp7InBhZ2UiOiJwcm9maWxlIiwicHJldmlvdXNQYWdlIjpudWxsLCJwb3NpdGlvbiI6InBhZ2VDb250ZW50In19)
