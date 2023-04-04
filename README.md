# AI models for automated segmentation of engineered polycystic kidney tubules
**Authors:** Monaco S., Bussola N., Buttò S., Sona D., Giobergia F., Jurman G., Xinaris C, Apiletti D.

Repository for the PyTorch implementation of the segmentation pipeline proposed in the paper.

## Abstract
Autosomal dominant polycystic kidney disease (ADPKD) is a monogenic, rare disease, characterized by the formation of multiple cysts that grow out of the renal tubules. Despite intensive attempts to develop new drugs or repurpose existing ones, there is currently no definitive cure for ADPKD. This is primarily due to the complex and variable pathogenesis of the disease and the lack of models that can faithfully reproduce the human phenotype.
Therefore, the development of models that allow automated detection and growth directly on human kidney tissue is a crucial step in the search for efficient therapeutic solutions.
Artificial Intelligence methods, and deep learning algorithms in particular, can provide powerful and effective solutions to such tasks, and indeed various architectures have been proposed in the literature in recent years.
Here, we comparatively review state-of-the-art deep learning segmentation models, using as a testbed a set of sequential RGB immunofluorescence images from 4 _in vitro_ experiments with 32 engineered polycystic kidney tubules.
To gain a deeper understanding of the detection process, we specialize the performance metrics used to evaluate the algorithms at both the pixel and cyst-wise levels.
Overall, two models stand out as the best performing, namely UNet++ and UACANet: the latter uses a self-attention mechanism introducing some explainability aspects that can be further exploited in future developments, thus making it the most promising algorithm to build upon towards a more refined cyst-detection platform.
UACANet models achieve a cyst-wise Intersection over Union of 0.83, 0.91 for Recall, and 0.92 for Precision when applied to detect large-size cysts. On all-size cysts, UACANet averages at 0.624 pixel-wise Intersection over Union. The code to reproduce all results is freely available in a public GitHub repository.

## Configuration and validation
Dependency can be installed using the following command:

```
pip install -r requirements.txt
```

You can run the code to train and evaluate the models by
```
python run.py
```
```
usage: run.py [-h] [-c CONFIG_PATH] [-d DATASET] [--tag TAG] [-e EXP] [--single_exp SINGLE_EXP] [-t TUBE] 
              [-m MODEL] [-l LOSS] [-k K] [-s SEED] [--stratify_fold [STRATIFY_FOLD]] [-f FOCUS_SIZE]
              [--tiling [TILING]] [--save_results [SAVE_RESULTS]] [--wb [WB]] [--evaluate_exp [EVALUATE_EXP]]

```

The script performs a sigle step of a LOTO cross validation (read the paper for more on this).

## Cite

This work has been submitted to Nature Scientific Reports
