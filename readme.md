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

Identifying named entities in a sentence is a common task in NLP pipelines. There are
an extensive set of datasets available online with marked entities such as famous person 
or geographic location but often more bespoke categories are needed for particular application.
An example of this could be a chatbot application which may begin processing a message
from a user by labelling problem specific relevant entities in their message. Often the process
of labelling this data must be done laboriously internally especially in a PoC phase. 
This results in there being only a small amount of labelled data with the potential addition 
of unlabelled data.

The aim behind this project is to design a solution to learn as much as possible from 
the small amount of labelled data without over-fitting as well as leveraging 
unlabelled data to improve generalisation to unseen data.

## Solution Components

### BERT Model

Learning representation for words on large corpora which are applicable across many NLP application has 
been an active area of research in the last decade. A large recent success came from Mikolov et al. [1] and Pennington et al. [2]
in using deep neural networks to produce the pre-train word embeddings Word2Vec and GloVe respectively.
These representation were hugely popular across the NLP field and gave a considerable boost in 
performance of the naive one-hot encoding approach especially when training data is limited.

More recently, a huge breakthrough in learned representation by Devlin et al. with the 
design of the BERT model [3]. The model comprises of 12 transformer layers and learns a 
representation for the context of a word in a given sentence. For different downstream tasks, 
minimal additional parameter are added and the whole model is fine tuned to the data. This
differs from the use of the pre-trained word embeddings such as GloVe which are kept static 
during optimisation in a downstream task.

The pre-trained BERT model has achieved state-of-the-art performance on a number of NLP
tasks seems like the most appropriate architecture for the NER problem especially when data 
is limited. However, fine tuning hundreds of millions of parameters requires considerable
computing power that was not available for this project. Lan et al. developed a lighter 
version of BERT called ALBERT by using factorising embedding parameterisation and 
cross-layer parameter sharing [4] but this was still considered to be too heavy. As a work
around, the pre-trained BERT layers were kept fixed with an additional MLP added to fine tune
the BERT output embeddings to solve the named entity recognition task. The precise architecture 
can be found in section X.X. 

### Confidence KL Regularizer

### Data Distribution KL Regularizer

## Data

## Implementation Details

### Precise Architecture

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

### Latent Space Representations

### Evaluation

## Future Work

## Install

## Run

## Library Structure

## References