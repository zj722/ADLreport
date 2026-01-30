# ADLS Reports 

## Lab 0

## Lab 1

## Lab 2
### Overview 
In Tutorial 5, we implemented the NAS(neural Architecture Search) using optuna with different search sampler including Grid, Random and TPE-based search. 

#### Task 1
In task 1, we assessed the accuracy and efficiency accross between TPE and Grid samplers by running NAS for 30 trials and plot the accuracy of best model structure against trials. The result is shwon below:



#### Task 2
The primary objective of this task is to obtain an efficient model that maintains high accuracy after compression (quantization and pruning). However, simply applying a compression pipeline sequentially after a standard NAS often yields suboptimal results. An architecture that performs best in its uncompressed state may be highly sensitive to quantization or pruning, whereas a slightly less accurate architecture might exhibit greater robustness against compression. To address this discrepancy, we evaluated and compared three different workflows:
1. **Standard NAS + Post Compression Pipeline**： In this baseline approach, the compression pipeline is applied only after the optimal architecture has been found by a standard NAS (from Task 1). Since the search phase does not account for compression, there is no guarantee that the selected architecture will retain its performance after quantization and pruning.     
2. **Compression-Aware Search (Without Post Compression Training)**: In this workflow, the compression steps are integrated directly into the search loop. For each trial, the model is constructed, trained, and immediately compressed before evaluation. This method aims to find architectures that are natively robust, meaning they maintain high accuracy immediately after compression without requiring further adaptation.

3. **Compression-Aware Search with Post-Compression Training**: Similar to the second approach, compression is applied within each trial. However, crucially, we perform additional training on the compressed model before final evaluation. This step aiming to recover accuracy lost during compression. This method seeks the global optimum by finding architectures that are not just robust, but also have high "recoverability" through fine-tuning.


**The comparison of result are shown below.**

Figure 1 illustrates the search trajectory of the three different workflows over 30 trials. The specific trends for each curve are analyzed below:

![Figure 1: Curves for the three workflows](imgs/nas_comparison_curve.png)

* **Curve 1: Baseline (Standard NAS without Compression)**
    Represented by the **blue curve**, this trajectory serves as the performance benchmark (FP32 accuracy). It remains relatively stable and high (~0.86) which is selected dut to the **TPESampler()** is verified to obtain the best result from task1. However, since this workflow ignores compression entirely, this curve represents a "theoretical upper bound" for uncompressed models, serving as a reference point to measure the impact of quantization and pruning in the other tasks.

* **Curve 2: Compression-Aware Search (Without Post-Compression Training)**
    The **orange curve** demonstrates the highest volatility. It starts with a significantly low accuracy (~0.53), revealing that unoptimized architectures have a high inherent sensitivity to compression without being retrained. However, the curve's rapid ascent proves that the search algorithm successfully identified architectures with high native robustness. Despite this improvement, the curve plateaus below the Baseline (Curve 1), confirming that relying solely on architectural robustness is insufficient to fully recover the accuracy lost during compression.

* **Curve 3: Compression-Aware Search (With Post-Compression Training)**
    The **green curve** represents the optimal workflow. It outperforms Curve 2, verifying that retraining is essential for recovering accuracy. Most notably, this curve eventually surpasses the Baseline (Curve 1), achieving the highest final accuracy (~0.87). This may caused by additional training epochs brought by retraining process. Also the combination of compression constraints and additional retraining may acted as a form of regularization, helping the model generalize better on the dataset than the standard FP32 model.


## Lab 3

## Lab 4


