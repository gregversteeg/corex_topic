import cPickle
import numpy as np
import sklearn
import sklearn.datasets
import re
import sys
from nltk.stem.snowball import *
stemmer = SnowballStemmer('english')

import scipy.sparse as ss
import sklearn.feature_extraction.text as skt
pattern = '\\b[A-Za-z]+\\b'

def chunks(doc, n=100):
    """Yield successive approximately equal and n-sized chunks from l."""
    words = doc.split()
    if len(words) == 0:
        yield ''
    n_chunks = len(words) / n  # Round down
    if n_chunks == 0:
        n_per_chunk = n
    else:
        n_per_chunk = int(np.ceil(float(len(words)) / n_chunks))  # round up
    for i in xrange(0, len(words), n_per_chunk):
        yield ' '.join(words[i:i+n])

def av_bbow(docs, n=100):
    # Average binary bag of words if we take chunks of a doc of size n
    proc = skt.CountVectorizer(token_pattern=pattern)
    proc.fit(docs)
    n_doc, n_words = len(docs), len(proc.vocabulary_)
    mat = ss.lil_matrix((n_doc, n_words))
    for l, doc in enumerate(docs):
        subdocs = chunks(doc, n=n)
        mat[l] = (proc.transform(subdocs) > 0).mean(axis=0).A.ravel()
    return mat.asformat('csr'), proc

def bow(docs):
    proc = skt.CountVectorizer(token_pattern=pattern)
    return proc.fit_transform(docs), proc

pruned = sklearn.datasets.fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))
if len(sys.argv) < 2:
    print 'python get_twenty.py 1000 \nTo get the 1000 most frequent words'
ntop = int(sys.argv[1])  # most frequent words

pruned.data = [unicode(doc).translate({ord(k):None for k in u"'"}) for doc in pruned.data]
pruned.data = [' '.join(stemmer.stem(w) for w in re.findall(pattern, doc)) for doc in pruned.data]

pruned_matrix, proc = av_bbow(pruned.data)
# pruned_matrix, proc = bow(pruned.data)

ns, nv = pruned_matrix.shape
print 'shape', ns, nv

var_order = np.argsort(- pruned_matrix.sum(axis=0).A1)[:ntop]
pruned_matrix = pruned_matrix[:, var_order]

#Dictionary
ivd = {v: k for k, v in proc.vocabulary_.items()}
words = [ivd[v] for v in var_order]
cPickle.dump(words, open('dictionary%d.dat'%ntop, 'w'),protocol=-1)
print ','.join(words)

#output matrices

# Eliminate empty docs?
# original_ids = np.where(pruned_matrix.sum(axis=1).A.ravel() > 0)[0]
# pruned_matrix = pruned_matrix[original_ids]
# cPickle.dump(original_ids, open('original_ids_%d.dat'%ntop, 'w'))

cPickle.dump(pruned_matrix, open('twenty_mat%d.dat'%ntop,'w'),protocol=-1)
cPickle.dump(ss.csr_matrix(pruned_matrix>0), open('twenty_mat%d_binary.dat'%ntop,'w'),protocol=-1)

