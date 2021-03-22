# Anchored CorEx: Hierarchical Topic Modeling with Minimal Domain Knowledge

**Cor**relation **Ex**planation (CorEx) is a topic model that yields rich topics that are maximally informative about a set of documents. The advantage of using CorEx versus other topic models is that it can be easily run as an unsupervised, semi-supervised, or hierarchical topic model depending on a user's needs. For semi-supervision, CorEx allows a user to integrate their domain knowledge via "anchor words."  This integration is flexible and allows the user to guide the topic model in the direction of those words. This allows for creative strategies that promote topic representation, separability, and aspects. More generally, this implementation of CorEx is good for clustering any sparse binary data.

If you use this code, please cite the following:

> Gallagher, R. J., Reing, K., Kale, D., and Ver Steeg, G. "[Anchored Correlation Explanation: Topic Modeling with Minimal Domain Knowledge](https://www.transacl.org/ojs/index.php/tacl/article/view/1244)." *Transactions of the Association for Computational Linguistics (TACL)*, 2017.

## Getting Started

### Install

Python code for the CorEx topic model can be installed via pip:

```
pip install corextopic
```

### Example Notebook

Full details on how to retrieve and interpret output from the CorEx topic model are given in the [example notebook](https://github.com/gregversteeg/corex_topic/blob/master/corextopic/example/corex_topic_example.ipynb). Below we describe how to get CorEx running as an unsupervised, semi-supervised, or hierarchical topic model.

### Running the CorEx Topic Model

Given a doc-word matrix, the CorEx topic model is easy to run. The code follows the scikit-learn fit/transform conventions.

```python
import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct

# Define a matrix where rows are samples (docs) and columns are features (words)
X = np.array([[0,0,0,1,1],
              [1,1,1,0,0],
              [1,1,1,1,1]], dtype=int)
# Sparse matrices are also supported
X = ss.csr_matrix(X)
# Word labels for each column can be provided to the model
words = ['dog', 'cat', 'fish', 'apple', 'orange']
# Document labels for each row can be provided
docs = ['fruit doc', 'animal doc', 'mixed doc']

# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=2)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=words, docs=docs)
```

Once the model is trained, we can get topics using the ```get_topics()``` function.

```python
topics = topic_model.get_topics()
for topic_n,topic in enumerate(topics):
    # w: word, mi: mutual information, s: sign
    topic = [(w,mi,s) if s > 0 else ('~'+w,mi,s) for w,mi,s in topic]
    # Unpack the info about the topic
    words,mis,signs = zip(*topic)    
    # Print topic
    topic_str = str(topic_n+1)+': '+', '.join(words)
    print(topic_str)
```

Similarly, the most probable documents for each topic can be accessed through the ``get_top_docs()`` function.

```python
top_docs = topic_model.get_top_docs()
for topic_n, topic_docs in enumerate(top_docs):
    docs,probs = zip(*topic_docs)
    topic_str = str(topic_n+1)+': '+', '.join(docs)
    print(topic_str)
```

Summary files and visualizations can be outputted from ```vis_topic.py```.

```python
from corextopic import vis_topic as vt
vt.vis_rep(topic_model, column_label=words, prefix='topic-model-example')
```


### Choosing the Number of Topics

Each topic explains a certain portion of the *total correlation* (TC). We can access the topic TCs through the `tcs` attribute, as well as the overall TC (the sum of the topic TCs) through the `tc` attribute. To determine how many topics we should use, we can look at the distribution of `tcs`.  If adding additional topics contributes little to the overall TC, then the topics already explain a large portion of the information in the documents. If this is the case, then we likely do not need more topics in our topic model. So, as a general rule of thumb continue adding topics until the overall TC plateaus.

We can also restart the CorEx topic model from several different initializations. This allows CorEx to explore different parts of the topic space and potentially find more informative topics. If we want to follow a strictly quantitative approach to choosing which of the multiple topic model runs we should use, then we can choose the topic model that has the highest TC (the one that explains the most information about the documents)




## Semi-Supervised Topic Modeling

### Using Anchor Words

Anchored CorEx allows a user integrate their domain knowledge through "anchor words." Anchoring encourages (but does not force) CorEx to search for topics that are related to the anchor words. This helps us find topics of interest, enforce separability of topics, and find aspects around topics.

If ```words``` is initialized, then it is easy to use anchor words:

```python
topic_model.fit(X, words=words, anchors=[['dog','cat'], 'apple'], anchor_strength=2)
```

This anchors "dog" and "cat" to the first topic, and "apple" to the second topic. The `anchor_strength` is the relative amount of weight given to an anchor word relative to all the other words. For example, if `anchor_strength=2`, then CorEx will place twice as much weight on the anchor word when searching for relevant topics. The `anchor_strength` should always be set above 1. The choice of `anchor_strength` beyond that depends on the size of the vocabulary and the task at hand. We encourage users to experiment with ```anchor_strength``` to find what is useful for their own purposes.  

If ```words``` is not initialized, we can anchor by specifying the column indices of the document-term matrix that we wish to anchor on. For example,

```python
topic_model.fit(X, anchors=[[0, 2], 1], anchor_strength=2)
```

anchors the words of columns 0 and 2 to the first topic, and word 1 to the second topic.

### Anchoring Strategies

There are a number of strategies we can use with anchored CorEx. Below we provide just a handful of examples.

1. *Anchoring a single set of words to a single topic*. This can help promote a topic that did not naturally emerge when running an unsupervised instance of the CorEx topic model. For example, we might anchor words like "snow," "cold," and "avalanche" to a topic if we supsect there should be a snow avalanche topic within a set of disaster relief articles.

```python
topic_model.fit(X, words=words, anchors=[['snow', 'cold', 'avalanche']], anchor_strength=4)
```

2. *Anchoring single sets of words to multiple topics*. This can help find different aspects of a topic that may be discussed in several different contexts. For example, we might anchor "protest" to three topics and "riot" to three other topics to understand different framings that arise from tweets about political protests.

```python
topic_model.fit(X, words=words, anchors=['protest', 'protest', 'protest', 'riot', 'riot', 'riot'], anchor_strength=2)
```

3. *Anchoring different sets of words to multiple topics.* This can help enforce topic separability if there appear to be "chimera" topics that are not well-separated. For example, we might anchor "mountain," "Bernese," and "dog" to one topic and "mountain," "rocky," and "colorado" to another topic to help separate topics that merge discussion of Bernese Mountain Dogs and the Rocky Mountains.

```python
topic_model.fit(X, words=words, anchors=[['bernese', 'mountain', 'dog'], ['mountain', 'rocky', 'colorado']], anchor_strength=2)
```

The [example notebook](https://github.com/gregversteeg/corex_topic/blob/master/corextopic/example/corex_topic_example.ipynb) details other examples of using anchored CorEx. We encourage domain experts to experiment with other anchoring strategies that suit their needs.

**Note:** when running unsupervised CorEx, the topics are returned and sorted according to how much total correlation they each explain. When running anchored CorEx, the topics are not sorted by total correlation, and the first *n* topics will correspond to the *n* anchored topics in the order given by the model input.



## Hierarchical Topic Modeling

### Building a Hierarchical Topic Model

For the CorEx topic model, topics are latent factors that can be expressed or not in each document. We can use the matrices of these topic expressions as input for another layer of the CorEx topic model, yielding a hierarchical topic model.

```python
# Train the first layer
topic_model = ct.Corex(n_hidden=100)
topic_model.fit(X)

# Train successive layers
tm_layer2 = ct.Corex(n_hidden=10)
tm_layer2.fit(topic_model.labels)

tm_layer3 = ct.Corex(n_hidden=1)
tm_layer3.fit(tm_layer2.labels)
```

Visualizations of the hierarchical topic model can be accessed through ```vis_topic.py```.

```python
vt.vis_hierarchy([topic_model, tm_layer2, tm_layer3], column_label=words, max_edges=300, prefix='topic-model-example')
```



## Technical notes

### Binarization of Documents

For speed reasons, this version of the CorEx topic model works only on binary data and produces binary latent factors. Despite this limitation, our work demonstrates CorEx produces coherent topics that are as good as or better than those produced by LDA for short to medium length documents. However, you may wish to consider additional preprocessing for working with longer documents. We have several strategies for handling text data.

0. Naive binarization. This will be good for documents of similar length and especially short- to medium-length documents.

1. Average binary bag of words. We split documents into chunks, compute the binary bag of words for each documents and then average. This implicitly weights all documents equally.

2. All binary bag of words. Split documents into chunks and consider each chunk as its own binary bag of words documents.This changes the number of documents so it may take some work to match the ids back, if desired. Implicitly, this will weight longer documents more heavily. Generally this seems like the most theoretically justified method. Ideally, you could aggregate the latent factors over sub-documents to get 'counts' of latent factors at the higher layers.

3. Fractional counts. This converts counts into a fraction of the background rate, with 1 as the max. Short documents tend to stay binary and words in long documents are weighted according to their frequency with respect to background in the corpus. This seems to work Ok on tests. It requires no preprocessing of count data and it uses the full range of possible inputs. However, this approach is not very rigorous or well tested.

For the python API, for 1 and 2, you can use the functions in ```vis_topic``` to process data or do the same yourself. Naive binarization is specified through the python api with count='binarize' and fractional counts with count='fraction'. While fractional counts may be work theoretically, their usage in the CorEx topic model has not be adequately tested.

### Single Membership of Words in Topics

Also for speed reasons, the CorEx topic model enforces single membership of words in topics. If a user anchors a word to multiple topics, the single membership will be overriden.


## References
If you use this code, please cite the following:

> Gallagher, R. J., Reing, K., Kale, D., and Ver Steeg, G. "[Anchored Correlation Explanation: Topic Modeling with Minimal Domain Knowledge](https://www.transacl.org/ojs/index.php/tacl/article/view/1244)." *Transactions of the Association for Computational Linguistics (TACL)*, 2017.

See the following papers if you interested in how CorEx works generally beyond sparse binary data.

> [*Discovering Structure in High-Dimensional Data Through Correlation Explanation*](http://arxiv.org/abs/1406.1222), Ver Steeg and Galstyan, NIPS 2014. <br>

>[*Maximally Informative Hierarchical Representions of High-Dimensional Data*](http://arxiv.org/abs/1410.7404), Ver Steeg and Galstyan, AISTATS 2015.
