# Detection of Counterfeit News using Machine Learning

## Objectives

* Build  models that can differentiate between Real news and Fake news
* Perform various text cleaning steps (Data preprocessing)
* Generate feature vectors using Doc2Vec and Word Embedding
* Classify the dataset using Naive Bayesian, Support Vector Machine, Neural network, Long short term machine
* Find the most accurate algorithm for fake news detection

## Scope

Develop fake news detection models with different machine learning algorithms to identify  the most accurate  algorithm for classification

## Data

The datasets used for this project were drawn from Kaggle. The training dataset has about 17600 rows of data from various articles on the internet.

1. id: unique id for a news article
2. title: the title of a news article
3. author: author of the news article
4. text: the text of the article; incomplete in some cases
5. label: a label that marks the article as potentially unreliable
           • 1: unreliable
           • 0: reliable

## Feature Extraction And Preprocessing
 * We perform some basic pre-processing of the data which produces a comma-separated list of words
* This can be input into the Doc2Vec algorithm and Word Embedding
* Doc2Vec algorithm will produce an 300-length embedding vector for each article
* Word Embedding is used to create vectors for LSTM 

# Results and Analysis
## Algorithm’s Performance

| Algorithm     | Accuracy      | Precision     | Recall        |F1-Score |
| ------------- | ------------- |---------------|---------------|---------|
| Naive Bayes   | 72            |      67       |  85           |  75     |
| SVM           | 88            |      84       |  93           |  88     |
| Neural Network| 90            |      86       |  94           |  89     |
| LSTM          | 91            |      96       |  92           |  95     |

## Conclusion

* The LSTM treats text as serialized objects. This makes LSTM far more efficient and accurate. 
* The major difference here is that the prediction made by LSTM considers not only the text but also the order of the text. 
* Therefore the most accurate algorithm for classification is LSTM

## Prediction of LSTM
Since LSTM is more accurate a LSTM model is build to preditc whether a given news is fake or real 








