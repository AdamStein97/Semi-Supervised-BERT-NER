# Semi-Supervised Named Entity Recognition with BERT and KL Regularizers

An exploration in using the pre-trained BERT model to perform Named Entity Recognition 
(NER) where labelled training data is limited but there is a considerable amount of unalabelled data.
Two different regularisation terms using Kullback–Leibler (KL) divergence are proposed that aim to 
leverage the unlabelled data to help the model generalise to unseen data.

### Models:
- __NER Baseline__: A simple approach to predicted named entity labels through a MLP
- __BERT NER__: Words are encoded using a static pre-trained BERT layer, a MLP on top 
predicts the named entity label
- __BERT NER Confidence KL__: Same architecture as `BERT NER` but in the final epochs of
training a KL term is introduced to encourage the model to have high confidence when 
predicting the unlabelled training examples
- __BERT NER Data Distribution KL__: Same architecture as `BERT NER` but in the final epochs of
training a KL term is introduced to encourage predicted labels for the unlabelled
data to match the expected probability distribution of the data.  

## Motivation

High Level motivation

### BERT Model

### Confidence KL Regularizer

### Data Distribution KL Regularizer

## Data

## Implementation Details

### Optimisation

### Tokenisation

### Masking
Mention epsilon divides

## Results

| Model Name  | Validation Accuracy  | Validation Accuracy no Other |  Validation Mean F1 | 
|---|---|---|---|
| NER Baseline  | 93.65% |  66.73% | 0.5457  |
| BERT NER | 95.90% | 78.41%  | 0.6099 | 
| BERT NER Confidence KL | __96.05%__  |  80.82% | __0.6514__  | 
| BERT NER Data Distribution KL  | 94.44%  | __83.50%__  | 0.6320 | 

### Masking

### Latent Space Representations

### Evaluation


## Future Work

## Install

## Run

## Library Structure