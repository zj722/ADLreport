# ADLS Reports 

## Lab 0

## Lab 1

## Lab 2
### Overview 
In Tutorial 5, we implemented the NAS(neural Architecture Search) using optuna with different search sampler including Grid, Random and TPE-based search. 

#### Task 1
In task 1, we assessed the accuracy and efficiency accross between TPE and Grid samplers by running NAS for 100 trials and plot the accuracy of best model structure against trials. The result is shwon below:



#### Task 2
The primary objective of this task is to obtain an efficient model that maintains high accuracy after compression (quantization and pruning). However, simply applying a compression pipeline sequentially after a standard NAS often yields suboptimal results. An architecture that performs best in its uncompressed state may be highly sensitive to quantization or pruning, whereas a slightly less accurate architecture might exhibit greater robustness against compression. To address this discrepancy, we evaluated and compared three different workflows:
1. **Standard NAS + Post Compression Pipeline**： In this baseline approach, the compression pipeline is applied only after the optimal architecture has been found by a standard NAS (from Task 1). Since the search phase does not account for compression, there is no guarantee that the selected architecture will retain its performance after quantization and pruning.     
2. **Compression-Aware Search (Without Post Compression Training)**: In this workflow, the compression steps are integrated directly into the search loop. For each trial, the model is constructed, trained, and immediately compressed before evaluation. This method aims to find architectures that are natively robust, meaning they maintain high accuracy immediately after compression without requiring further adaptation.

3. **Compression-Aware Search with Post-Compression Training**: Similar to the second approach, compression is applied within each trial. However, crucially, we perform additional training on the compressed model before final evaluation. This step aiming to recover accuracy lost during compression. This method seeks the global optimum by finding architectures that are not just robust, but also have high "recoverability" through fine-tuning.


**The comparison of result are shown below.**



## Lab 3

## Lab 4


