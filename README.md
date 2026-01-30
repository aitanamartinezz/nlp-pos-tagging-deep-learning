# Multilingual Part-of-Speech Tagging with LSTM Networks

This repository contains the implementation of neural Part-of-Speech (POS) tagging models
using Long Short-Term Memory (LSTM) networks for English, Spanish, and French. The project
is based on Universal Dependencies datasets and focuses on sequence labeling using deep
learning techniques.

---

## Project Description

Part-of-Speech tagging is a fundamental task in Natural Language Processing (NLP) that
consists of assigning a grammatical category (noun, verb, adjective, etc.) to each word
in a sentence. This project addresses POS tagging as a sequence labeling problem using
neural networks.

The objective of the project is to design, train, and evaluate LSTM-based models for
multilingual POS tagging, comparing different architectural configurations and analyzing
their performance across languages.

The models are trained and evaluated on English, Spanish, and French datasets, achieving
high accuracy in all cases.

---

## Dataset

The datasets used in this project are obtained from the **Universal Dependencies (UD)**
project and are provided in **CoNLL-U format**.

Languages included:
- English
- Spanish
- French

Each dataset consists of sentences annotated with Universal POS tags. The notebook
automatically downloads and processes the required datasets.

---

## Repository Structure

NLUPOSTAGGING.ipynb # Main notebook containing the full implementation
README.md # Project documentation


---

## Requirements and Dependencies

The project is implemented in **Python** and relies on the following libraries:

- TensorFlow
- Keras
- NumPy
- Pandas
- conllu
- tqdm
- ipykernel

All experiments are executed within a Jupyter Notebook environment.

---

## Usage Instructions

### Running the Project

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook NLUPOSTAGGING.ipynb

2. Run the notebook sequentially from top to bottom.

The notebook is self-contained and performs the following steps automatically:

 - Downloads the Universal Dependencies datasets
 - Preprocesses and encodes the data
 - Trains multiple LSTM model configurations
 - Evaluates model performance
 - Tests predictions on example sentences


## Data Preprocessing

The preprocessing pipeline includes:

- Parsing CoNLL-U formatted files  
- Filtering sentences longer than 128 tokens  
- Building vocabularies for words and POS tags  
- Integer encoding of tokens and labels  
- Padding sequences to a fixed length  
- Handling out-of-vocabulary tokens using a special `<UNK>` token  

---

## Model Architecture

The neural architecture consists of the following components:

### Embedding Layer
- Learnable word embeddings  
- Masking applied to padding tokens  

### LSTM Layer
- Unidirectional or Bidirectional LSTM  
- Dropout regularization  

### Output Layer
- TimeDistributed dense layer  
- Softmax activation for POS tag prediction  

### Custom Metric
- Masked accuracy that ignores padding positions  

---

## Training Configuration

Multiple configurations were evaluated by varying:

- Embedding dimension  
- Hidden layer dimension  
- Dropout rate  
- Bidirectionality  
- Optimizer (Adam or RMSprop)  

### Training Parameters
- **Epochs:** 5  
- **Batch size:** 32  

The best model for each language was selected based on validation accuracy.

---

## Results

### Test Accuracy
- **English:** 93.73%  
- **Spanish:** 94.07%  
- **French:** 95.16%  

### Best Model Configurations

| Language | Configuration | Parameters | Test Accuracy |
|---------|---------------|------------|---------------|
| English | (64,128,0.3,Bi,Adam) | 781K | 93.73% |
| Spanish | (64,128,0.3,Bi,Adam) | 1.33M | 94.07% |
| French | (128,256,0.3,Bi,Adam) | 2.84M | 95.16% |

**Configuration format:**  
`(embedding_dim, hidden_dim, dropout, bidirectional, optimizer)`

---

## Analysis and Observations

- Bidirectional LSTM models consistently outperform unidirectional models by approximately  
  2–2.5% across all languages.
- Adam optimizer provides better performance than RMSprop in all tested configurations.
- Smaller embedding sizes achieve competitive performance with significantly fewer parameters.
- French achieves the highest accuracy, followed by Spanish and English, correlating with  
  dataset characteristics and morphological structure.

---

## Error Analysis

Qualitative analysis of model predictions highlights:

- Errors related to capitalization at sentence beginnings  
- Difficulties handling contractions in Romance languages  
- Morphological ambiguity across languages  
- Out-of-vocabulary words mapped to the `<UNK>` token  

---

## Conclusion

This project demonstrates the effectiveness of LSTM-based neural networks for multilingual
POS tagging. The results confirm the importance of bidirectional architectures and proper
hyperparameter selection in sequence labeling tasks.

The implementation provides a solid foundation for further experimentation with more
advanced architectures such as Transformers or contextual embeddings.

---

## Authors

- Aitana Martínez Rey  
- Marina Sánchez Villaverde  
- Pablo Soage Rodas  

---

## References

- Universal Dependencies: https://universaldependencies.org  
- CoNLL-U Format  
- Universal POS Tags  
