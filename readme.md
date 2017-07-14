# Anchored CorEx: Hierarchical Topic Modeling with Minimal Domain Knowledge

**Contributors:** [Greg Ver Steeg](https://www.isi.edu/people/gregv/about)<sup>1</sup>, 
[Ryan J. Gallagher](http://ryanjgallagher.github.io/)<sup>1,2</sup>, 
[David Kale](http://www-scf.usc.edu/~dkale/)<sup>1</sup>

<sup>1</sup>[Information Sciences Institute](https://www.isi.edu/), University of Southern California, 
<sup>2</sup>[Network Science Institute](https://www.networkscienceinstitute.org/), Northeastern University

## Overview

The principle of *Cor*-relation *Ex*-planation has recently been introduced as a way to build rich representations that
are maximally informative about data. This project optimizes the CorEx framework for sparse binary data, so that it can be leveraged for topic modeling. Our work demonstrates CorEx finds coherent, meaningful topics that are competitive with LDA topics across a variety of metrics, despite the fact CorEx only utilizes binary counts.

This code also introduces an anchoring mechanism for integrating the CorEx topic model with domain knowledge via the information bottleneck. This anchoring is flexible and allows the user to anchor multiple words to one topic, one word to multiple topics, or any other creative combination in order to uncover topics that do not naturally emerge.

Detailed analysis and applications of the CorEx topic model using this code:<br>
[*Anchored Correlation Explanation: Topic Modeling with Minimal Domain Knowledge*](https://arxiv.org/abs/1611.10277), Gallagher et al., preprint 2017.

Underlying motivation and theory of CorEx:<br>
[*Discovering Structure in High-Dimensional Data Through Correlation Explanation*](http://arxiv.org/abs/1406.1222), Ver Steeg and Galstyan, NIPS 2014. <br>
[*Maximally Informative Hierarchical Representions of High-Dimensional Data*](http://arxiv.org/abs/1410.7404), Ver Steeg and Galstyan, AISTATS 2015.

This code can be used for any sparse binary dataset. In principle, continuous values in the range zero to one can also be used as 
inputs but the effect of this is not well tested. 

### Install

To install, download using [this link](https://github.com/gregversteeg/corex_topic/archive/master.zip) 
or clone the project by executing this command in your target directory:
```
git clone https://github.com/gregversteeg/corex_topic.git
```
Use *git pull* to get updates. 

The code is under development. Please contact Greg Ver Steeg about issues with this pre-alpha version.  

### Dependencies

CorEx requires numpy and scipy. If you use OS X, we recommend installing the [Scipy Superpack](http://fonnesbeck.github.io/ScipySuperpack/).

The visualization capabilities in vis_topic.py require other packages: 
* matplotlib - Already in scipy superpack.
* [networkx](http://networkx.github.io)  - A network manipulation library. 
* sklearn - Already in scipy superpack and only required for visualizations. 
* seaborn - Only required for visualizations
* [graphviz](http://www.graphviz.org) (optional, for compiling produced .dot files into pretty graphs. The command line 
tools are called from vis_topic. Graphviz should be compiled with the triangulation library for best visual results).

## Running the CorEx Topic Model

Given a doc-word matrix, the CorEx topic model is easy to train. The code follows the scikit-learn fit/transform conventions.

```python
import corex_topic as ct
import vis_topic as vt
import scipy.sparse as ss

# Define a matrix where rows are samples (docs) and columns are features (words)
X = np.array([[0,0,0,1,1],
              [1,1,1,0,0],
              [1,1,1,1,1]], dtype=int)
# Sparse matrices are also supported 
X = ss.csr_matrix(X)
# Word labels for each column can be provided to the model
words = ['dog', 'cat', 'fish', 'apple', 'orange']

# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=2)  # Define the number of latent (hidden) topics to use.
topic_model.fit(X, words=words)
```

Once the model is trained, the topics can be accessed through the ```get_topics()``` function.

```python
topic_model.get_topics()
```

Summary files and visualizations can be outputted from ```vis_topic.py```.

```python
vt.vis_rep(topic_model, column_label=words, prefix='topic-model-example')
```

The [corex-topic-example notebook](https://github.com/gregversteeg/corex_topic/blob/master/examples/corex-topic-example.ipynb) gives full details and examples on how to retrieve and interpret output from the CorEx topic model.


## Building a Hierarchical Topic Models

It is natural and straightforward to extend the CorEx topic model to a hierarchical representation.

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
Each topic explains a certain portion of the *total correlation*. These topic TCs can be accessed through the ```tcs``` attribute, and the overall TC (the sum of the topic TCs) can be accessed through ```tc```. To assess how many topics to choose at each layer, you may look at the distribution of ```tcs``` for each layer. As a rule of thumb, additional latent topics should be added until additional topics contribute little (less than 1%) to the overall TC.

Visualizations of the hierarchical topic model can be accessed through ```vis_topic.py``` if you have graphviz installed.

```python
vt.vis_hierarchy([topic_model, tm_layer2, tm_layer3], column_label=words, max_edges=300, prefix='topic-model-example')
```

To get better topic results, you can restart the CorEx topic model several times from different initializations, and choose the topic model that has the highest TC (explains the most information about the documents).


## Anchoring for Semi-Supervised Topic Modeling

Anchored CorEx allows a user to anchor words to topics in a semi-supervised fashion to uncover otherwise elusive topics. If ```words``` is initialized, anchoring is effortless:

```python
topic_model.fit(X, words=words, anchors=[['dog','cat'], 'apple'], anchor_strength=2)
```

This anchors "dog" and "cat" to the first topic, and "apple" to the second topic. As a rule of thumb ```anchor_strength``` should always be set above 1, where setting ```anchor_strength``` between 1 and 3 gently nudges a topic towards the anchor words, and setting it above 5 more strongly encourages the topic towards the anchor words. We encourage users to experiment with ```anchor_strength``` for their own purposes.

One word can be anchored to multiple topics, multiple words anchored to one topic, or any other combination of anchoring strategies. The [corex-topic-example notebook](https://github.com/gregversteeg/corex_topic/blob/master/examples/corex-topic-example.ipynb) details several strategies for anchoring.

If ```words``` is not initialized, you may anchor by specifying the integer column feature indices that you wish to anchor on.



## Technical notes

For speed reasons, this version of the CorEx topic model works only on binary data and produces binary latent factors. Despite this limitation, our work demonstrates CorEx produces coherent topics that are as good as or better than those produced by LDA for short to medium length documents. However, you may wish to consider additional preprocessing for working with longer documents. We have several strategies for handling text data. 
 
0. Naive binarization. This will be good for documents of similar length and especially short- to medium-length documents. 
 
1. Average binary bag of words. We split documents into chunks, compute the binary bag of words for each documents and then average. This implicitly weights all documents equally. 
                        
2. All binary bag of words. Split documents into chunks and consider each chunk as its own binary bag of words documents. 
 This changes the number of documents so it may take some work to match the ids back, if desired. Implicitly, this
 will weight longer documents more heavily. Generally this seems like the most theoretically justified method. Ideally, you could aggregate the latent factors over sub-documents to get 'counts' of latent factors at the higher layers. 
 
 3. Fractional counts. This converts counts into a fraction of the background rate, with 1 as the max. Short documents tend to stay binary and words in long documents are weighted according to their frequency with respect to background in the corpus. This seems to work Ok on tests. It requires no preprocessing of count data and it uses the full range of possible inputs. However, this approach is not very rigorous or well tested.
                        
For the python API, for 1 and 2, you can use the functions in ```vis_topic.py``` to process data or do the same yourself. Naive binarization is specified through the python api with count='binarize' and fractional counts with count='fraction'. While fractional counts may be work theoretically, their usage in the CorEx topic model has not be adequately tested.

Also note that also for speed reasons, the CorEx topic model enforces single membership of words in topics.
