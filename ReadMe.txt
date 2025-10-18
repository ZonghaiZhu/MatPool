# Matrix-pattern-oriented Pooling for Graph-level Representation and Prediction

## Overview

This work proposes a Matrix-pattern-oriented Pooling (MatPool), including Matrix Representation and Matrix Prediction. Matrix Representation primarily constructs a matrix-based feature representation of graph-structured data from the perspective of linear space isomorphism, aiming to preserve graph information. Matrix Prediction, through bilinear mapping, effectively extracts feature information by leveraging the row-column correlations inherent in matrix-based features and predicts the label of the entire graph. Theoretical analysis highlights the ability of Matrix Representation to preserve graph information and the strength of Matrix Prediction in handling matrix-based features. Extensive experiments across diverse graph property prediction benchmarks demonstrate that MatPool outperforms existing methods in both efficiency and effectiveness, offering a different paradigm for graph representation and prediction.

##
The document contains MatPool and MatPool_10folds. MatPool contains the codes to reproduce the results in Table 6 and MatPool-10folds contains the codes to reproduce the results in Table 7.

## Results

We have provided all classification results in the uploaded codes. Please check the files named 'log' to get the classification results.

All results can be reproduced by running main.py. In main.py, you can select the dataset and adjust the parameters. The main.py will use model.py which contains the fit function and predict function. We have used the grid search to conduct the experiment and record the results on validation and test data. The parameters combination that achieves the highest results on the validation data is selected to predict the test data. 


## Dependencies

* python >= 3.9
* pytorch >= 1.11.0
* torch_geometric >= 2.0.4
* numpy >= 1.21.5
* scikit-learn >= 1.0.2
* scipy >= 1.7.3

## Run Program

* main.py calls the main function to run the program
* model.py represents the model and contains training and testing
* net.py is the structure of the used neural network
* utils.py provides some tools to assist program execution 

Run main.py directly to get the results on validation and test data. You can change the parameters and choose different data set in argparse.ArgumentParser() to see different results. 

## Datasets

All datasets used in this work can be downloaded from OGBG datasets downloaded from "https://ogb.stanford.edu/docs/nodeprop/", and TUDataset downloaded from "https://chrsmrrs.github.io/datasets/docs/datasets/".

## Customization

### How to Prepare Your Own Dataset?

You can prepare the graph data like provided in pytorch_geometric



