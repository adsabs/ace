# Astrophysics Concept Extraction

Extract concepts or keywords from the ADS document corpus using simple or sophisticated methodologies.

## Minimal Overview

Ace is a simple implementation of a multi-label classifier. Under the hood, it is using scikit-learn.

#### Methodology

The following steps are applied:

 1. Application receives the full text of an articleNick got back to me, 
 2. The text has stop-words removed, is lemmatized, and then converted into a matrix
 3. The known keyword for this text is loaded, and converted into a vector
 4. An iteration of the classifier fitting routine is carried out

#### Classifier

Currently, a simple classifier is being used: logit

Each label is considered independent from one another, and has its own classifier. Logistic regression (or maximising the likelihood) is being carried out by stochastic gradient descent.

The drive for doing this was based on the classifiers available from `scikit-learn` that provide incremental fitting methods (if even feasible for that type of algorithm).

#### Overfitting, underfitting, and optimisations

The pipeline currently creates simple tools to investigate how well the classifier is functioning, and provides:
  * Confusion matrices (per label)
  * Metrics, such as precision and recall