# Astrophysics concept extraction: Ace

A brief specification and description of how we want to extract *concepts*, *keywords*, *sentiment*, and other semantic features from the full-text content of the Astrophysics Data System document corpus.

# Requirements

  * Document should be classified by UAT labels
  * Concept extraction of a single document should be **fast** so re-indexing is not slow
  * Supervised learning (see later): rebuilding of the model should be **reasonably fast**
  * Scalable to large corpus, without worrying about memory/CPU usage
  * Diagnostics of the chosen concepts to a document (ie., best-fit accessment)

# Architecture

The following architecture would be planned:

  1. Built and deployed within a Docker container (internally or on AWS)
  2. A web-service or binary that returns the *best-fit* concepts when given a document (or bibcode, or some identifier)
  3. If a supervised model is required, this is done separately to the run time environment, and the model is stored permanently on disk for fast access

# Extraction methods

I have been looking at two methods, that have varying advantages and disadvantages, and I'll describe them here

## Document "defined" keywords

The method applied by the latest classifier of Invenio uses a very simple algorithm: https://github.com/inveniosoftware/invenio-classifier. This works by finding the frequency of terms within the subject area's thesaurus is used (including their alternative and preferred labels). The ranking of the frequency (ie., the more it is used) results in the top N keywords.

**Advantages**:
  1. "Fast" and "scalable" - uses regex search on a document-by-document basis, so it is limited by the number of entries in the thesaurus and the length of the document
  2. Does not require any training data - ie., no one has to manually label any documents
  3. Simple - this usually means easier to maintain, extend, and troubleshoot

**Disadvantages**:
  1. The document **must** use some keyword known to the thesaurus, otherwise it will never be labelled with the correct concept, ie., limited by the words used by the author
  2. This would require two thesuari to be used/managed: astronomy and physics

## Supervised learning keywords

Another option, similar to KEA, is to use a *supervised learning* algorithm. This requires a set of documents to be curated by hand and labelled with UAT keywords (training data). A model is then fit with the training data, and the keywords for an unknown document are predicted using this model.

**Advantages**:
  1. Does not require the keyword tied to a thesaurus concept to have been defined, which in principle can result in more documents being labelled than those that rely on the words used by an author
  2. Easier to measure good/bad concept allocation, and to investigate what needs to be changed to improve the decisions made
  3. Scalable in terms of memory and CPU usage

**Disadvantages**:
  1. Requires training data, ie., someone has to manually annotate keywords for a set of documents
  2. Number of documents to annotate (learn from) depends upon how good the model used is (could range from 1-100% of our corpus)
  3. Learning data set needs to cover a range of content and thesaurus keywords to ensure that predictions are reliable
  4. More complex implimentation than above

# Supervised Learning

The system would work in the following manner:

  1. Select a range of papers that are to be used as our training data, and annotate them with keywords (for now, say they are concepts from the UAT)
  2. Incrementally load each training document and do the following:
    i. Obtain all words, remove stop words, lower case, and stem to lemmas
    ii. Convert the bag-of-words into a vector format, ie., for our entire corpus, we will have a dictionary of ID -> word, where each bag-of-words will have a spase matrix that has ID and number of occurences
    iii. Convert the spare-matrix of bag-of-words into a term frequency-inverse document frequency sparse-matrix
  3. Load the keywords we expect to be found for a document, and do the same processing as we did for the document
  4. Incrementally fit a *classifying algorithm* using *SVM* or *Naive Bayes* where the document vecetor is *X* and the keywords are *Y*.
  5. Save the fit model to disk

One can then simply pass a document through the fit to get a prediction for the concepts that describe it. This also allows one to easily test:

  * How good are the fits, using a set of *testing* documents
  * How the fits change based on number of *training* documents
  * How the fits change based on stop-words, lemmas, and other parsing options
  * How the fits change on keyword coverage

These steps are independent of memory. Using the package <a href="http://radimrehurek.com/gensim/">gensim</a>, it already provides the framework to load documents as strems, convert them to easily storable sparse-matrices. One can then use (or implement within gensim) the fitting of the model using partial fitting routines, that are natively available from scikit-learn.

If we are CPU bound, *gensim* also works natively with Pyro for distributed running.


# Reading List
 * Gensim: http://radimrehurek.com/gensim/
 * Invenio Classifier: https://github.com/inveniosoftware/invenio-classifier
 * Scikit-learn Mulitlabel Classification: http://scikit-learn.org/stable/auto_examples/plot_multilabel.html#example-plot-multilabel-py
 * TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
 * Bimodal example using Naive Bayes: http://radimrehurek.com/data_science_python/
 * Scikit-learn Label binarizer: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html

