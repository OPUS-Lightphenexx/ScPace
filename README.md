# ScPace: Timestamp Calibration for time-series ScRNA-seq expression data

> **Timestamp Calibration for time-series single cell RNA-seq expression data**<br>
>  Journal of Molecular Biology 2025<br>

> **Abstract:** 
>
> Timestamp automatic annotation (TAA) is a crucial procedure for analyzing time-series
 ScRNA-seq data, as they unveil dynamic biological developments and cell
 regeneration process. However, current TAA methods heavily rely on manual
 timestamps, often overlooking their reliability. This oversight can significantly degrade
 the performance of timestamp automatic annotation due to noisy timestamps.
 Nevertheless, the current approach for addressing this issue tends to select less critical
 cleaned samples for timestamp calibration. To tackle this challenge, we have
 developed a novel timestamp calibration model called ScPace for handling noisy
 labeled time-series ScRNA-seq data. This approach incorporates a latent variable
 indicator within a base classifier instead of probability sampling to detect noisy samples
 effectively. To validate our proposed method, we conducted experiments on both
 simulated and real time-series ScRNA-seq datasets. Cross-validation experiments with
 different artificial mislabeling rates demonstrate that ScPace outperforms previous
 approaches. Furthermore, after calibrating the timestamps of the original time-series
 ScRNA-seq data using our method, we performed supervised pseudotime analysis,
 revealing that ScPace enhances its performance significantly. These findings suggest
 that ScPace is an effective tool for timestamp calibration by enabling reclassification
 and deletion of detected noisy labeled samples while maintaining robustness across
 diverse ranges of time-series ScRNA-seq datasets.


We present ScPace, a novel approach for timestamp calibration in time-series single-cell RNA sequencing (scRNA-seq) data. This method tackles the significant issue of noisy timestamps in time-series ScRNA-seq data, which can undermine the accuracy of timestamp automatic annotation (TAA) and interfere with downstream analyses, such as supervised pseudotime analysis. ScPace leverages a latent variable indicator within a support vector machine (SVM) framework to efficiently identify and correct mislabeled samples, improving the robustness and reliability of results across a range of time-series ScRNA-seq datasets.

![graph_abstract](https://github.com/user-attachments/assets/ed7bbfb3-6c54-4109-97fb-916777b8f6ae)

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
- Arxiv: [Timestamp calibration for time-series single cell RNA-seq expression data](https://arxiv.org/abs/2412.03027)
- Researchgate: [Timestamp calibration for time-series single cell RNA-seq expression data](https://www.researchgate.net/publication/386426329_Timestamp_calibration_for_time-series_single_cell_RNA-seq_expression_data)
- Supplementary files: [Supplementary](https://www.researchgate.net/publication/389051068_SupplementaryScPacedocx?_tp=eyJjb250ZXh0Ijp7InBhZ2UiOiJwcm9maWxlIiwicHJldmlvdXNQYWdlIjpudWxsLCJwb3NpdGlvbiI6InBhZ2VDb250ZW50In19)
