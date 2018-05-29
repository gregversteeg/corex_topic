"""CorEx Hierarchical Topic Models

Use the principle of Total Cor-relation Explanation (CorEx) to construct
hierarchical topic models. This module is specially designed for sparse count
data and implements semi-supervision via the information bottleneck.

Greg Ver Steeg and Aram Galstyan. "Maximally Informative Hierarchical
Representations of High-Dimensional Data." AISTATS, 2015.

Gallagher et al. "Anchored Correlation Explanation: Topic Modeling with Minimal
Domain Knowledge." TACL, 2017.

Code below written by:
Greg Ver Steeg (gregv@isi.edu)
Ryan J. Gallagher
David Kale
Lily Fierro

License: Apache V2
"""

import numpy as np  # Tested with 1.8.0
from os import makedirs
from os import path
from scipy.misc import logsumexp  # Tested with 0.13.0
import scipy.sparse as ss
from six import string_types # For Python 2&3 compatible string checking
from sklearn.externals import joblib


class Corex(object):
    """
    Anchored CorEx hierarchical topic models
    Code follows sklearn naming/style (e.g. fit(X) to train)

    Parameters
    ----------
    n_hidden : int, optional, default=2
        Number of hidden units.

    max_iter : int, optional
        Maximum number of iterations before ending.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode. 1 outputs TC(X;Y) as you go
        2 output alpha matrix and MIs as you go.

    tree : bool, default=True
        In a tree model, each word can only appear in one topic. tree=False is not yet implemented.

    count : string, {'binarize', 'fraction'}
        Whether to treat counts (>1) by directly binarizing them, or by constructing a fractional count in [0,1].

    seed : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    labels : array, [n_samples, n_hidden]
        Label for each hidden unit for each sample.

    clusters : array, [n_visible]
        Cluster label for each input variable.

    p_y_given_x : array, [n_samples, n_hidden]
        p(y_j=1|x) for each sample.

    alpha : array-like, shape [n_hidden, n_visible]
        Adjacency matrix between input variables and hidden units. In range [0,1].

    mis : array, [n_hidden, n_visible]
        Mutual information between each (visible/observed) variable and hidden unit

    tcs : array, [n_hidden]
        TC(X_Gj;Y_j) for each hidden unit

    tc : float
        Convenience variable = Sum_j tcs[j]

    tc_history : array
        Shows value of TC over the course of learning. Hopefully, it is converging.

    words : list of strings
        Feature names that label the corresponding columns of X

    References
    ----------

    [1]     Greg Ver Steeg and Aram Galstyan. "Discovering Structure in
            High-Dimensional Data Through Correlation Explanation."
            NIPS, 2014. arXiv preprint arXiv:1406.1222.

    [2]     Greg Ver Steeg and Aram Galstyan. "Maximally Informative
            Hierarchical Representations of High-Dimensional Data"
            AISTATS, 2015. arXiv preprint arXiv:1410.7404.

    """

    def __init__(self, n_hidden=2, max_iter=200, eps=1e-5, seed=None, verbose=False, count='binarize',
                 tree=True, **kwargs):
        self.n_hidden = n_hidden  # Number of hidden factors to use (Y_1,...Y_m) in paper
        self.max_iter = max_iter  # Maximum number of updates to run, regardless of convergence
        self.eps = eps  # Change to signal convergence
        self.tree = tree
        np.random.seed(seed)  # Set seed for deterministic results
        self.verbose = verbose
        self.t = 20  # Initial softness of the soft-max function for alpha (see NIPS paper [1])
        self.count = count  # Which strategy, if necessary, for binarizing count data
        if verbose > 0:
            np.set_printoptions(precision=3, suppress=True, linewidth=200)
            print('corex, rep size:', n_hidden)
        if verbose:
            np.seterr(all='warn')
            # Can change to 'raise' if you are worried to see where the errors are
            # Locally, I "ignore" underflow errors in logsumexp that appear innocuous (probabilities near 0)
        else:
            np.seterr(all='ignore')

    def label(self, p_y_given_x):
        """Maximum likelihood labels for some distribution over y's"""
        return (p_y_given_x > 0.5).astype(bool)

    @property
    def labels(self):
        """Maximum likelihood labels for training data. Can access with self.labels (no parens needed)"""
        return self.label(self.p_y_given_x)

    @property
    def clusters(self):
        """Return cluster labels for variables"""
        return np.argmax(self.alpha, axis=0)

    @property
    def sign(self):
        """Return the direction of correlation, positive or negative, for each variable-latent factor."""
        return np.sign(self.theta[3] - self.theta[2]).T

    @property
    def tc(self):
        """The total correlation explained by all the Y's.
        """
        return np.sum(self.tcs)

    def fit(self, X, anchors=None, anchor_strength=1, words=None, docs=None):
        """
        Fit CorEx on the data X. See fit_transform.
        """
        self.fit_transform(X, anchors=anchors, anchor_strength=anchor_strength, words=words, docs=docs)
        return self

    def fit_transform(self, X, anchors=None, anchor_strength=1, words=None, docs=None):
        """Fit CorEx on the data

        Parameters
        ----------
        X : scipy sparse CSR or a numpy matrix, shape = [n_samples, n_visible]
            Count data or some other sparse binary data.

        anchors : A list of variables anchor each corresponding latent factor to.

        anchor_strength : How strongly to weight the anchors.

        words : list of strings that label the corresponding columns of X

        docs : list of strings that label the corresponding rows of X

        Returns
        -------
        Y: array-like, shape = [n_samples, n_hidden]
           Learned values for each latent factor for each sample.
           Y's are sorted so that Y_1 explains most correlation, etc.
        """
        X = self.preprocess(X)
        self.initialize_parameters(X, words, docs)
        if anchors is not None:
            anchors = self.preprocess_anchors(list(anchors))
        p_y_given_x = np.random.random((self.n_samples, self.n_hidden))
        if anchors is not None:
            for j, a in enumerate(anchors):
                p_y_given_x[:, j] = 0.5 * p_y_given_x[:, j] + 0.5 * X[:, a].mean(axis=1).A1  # Assumes X is a binary matrix

        for nloop in range(self.max_iter):
            if nloop > 1:
                for j in range(self.n_hidden):
                    if self.sign[j, np.argmax(self.mis[j])] < 0:
                        # Switch label for Y_j so that it is correlated with the top word
                        p_y_given_x[:, j] = 1. - p_y_given_x[:, j]
            self.log_p_y = self.calculate_p_y(p_y_given_x)
            self.theta = self.calculate_theta(X, p_y_given_x, self.log_p_y)  # log p(x_i=1|y)  nv by m by k

            if nloop > 0:  # Structure learning step
                self.alpha = self.calculate_alpha(X, p_y_given_x, self.theta, self.log_p_y, self.tcs)
            if anchors is not None:
                for a in flatten(anchors):
                    self.alpha[:, a] = 0
                for ia, a in enumerate(anchors):
                    self.alpha[ia, a] = anchor_strength

            p_y_given_x, _, log_z = self.calculate_latent(X, self.theta)

            self.update_tc(log_z)  # Calculate TC and record history to check convergence
            self.print_verbose()
            if self.convergence():
                break

        if self.verbose:
            print('Overall tc:', self.tc)

        if anchors is None:
            self.sort_and_output(X)
        self.p_y_given_x, self.log_p_y_given_x, self.log_z = self.calculate_latent(X, self.theta)  # Needed to output labels
        self.mis = self.calculate_mis(self.theta, self.log_p_y)  # / self.h_x  # could normalize MIs
        return self.labels

    def transform(self, X, details=False):
        """
        Label hidden factors for (possibly previously unseen) samples of data.
        Parameters: samples of data, X, shape = [n_samples, n_visible]
        Returns: , shape = [n_samples, n_hidden]
        """
        X = self.preprocess(X)
        p_y_given_x, _, log_z = self.calculate_latent(X, self.theta)
        labels = self.label(p_y_given_x)
        if details == 'surprise':
            # TODO: update
            # Totally experimental
            n_samples = X.shape[0]
            alpha = np.zeros((self.n_hidden, self.n_visible))
            for i in range(self.n_visible):
                alpha[np.argmax(self.alpha[:, i]), i] = 1
            log_p = np.empty((2, n_samples, self.n_hidden))
            c0 = np.einsum('ji,ij->j', alpha, self.theta[0])
            c1 = np.einsum('ji,ij->j', alpha, self.theta[1])  # length n_hidden
            info0 = np.einsum('ji,ij->ij', alpha, self.theta[2] - self.theta[0])
            info1 = np.einsum('ji,ij->ij', alpha, self.theta[3] - self.theta[1])
            log_p[1] = c1 + X.dot(info1)  # sum_i log p(xi=xi^l|y_j=1)  # Shape is 2 by l by j
            log_p[0] = c0 + X.dot(info0)  # sum_i log p(xi=xi^l|y_j=0)
            surprise = [-np.sum([log_p[labels[l, j], l, j] for j in range(self.n_hidden)]) for l in range(n_samples)]
            return p_y_given_x, log_z, np.array(surprise)
        elif details:
            return p_y_given_x, log_z
        else:
            return labels

    def predict_proba(self, X):
        return self.transform(X, details=True)

    def predict(self, X):
        return self.transform(X, details=False)

    def preprocess(self, X):
        """Data can be binary or can be in the range [0,1], where that is interpreted as the probability to
        see this variable in a given sample"""
        if X.max() > 1:
            if self.count == 'binarize':
                X = (X > 0)
            elif self.count == 'fraction':
                X = X.astype(float)
                count = np.array(X.sum(axis=0), dtype=float).ravel()
                length = np.array(X.sum(axis=1)).ravel().clip(1)
                bg_rate = ss.diags(float(X.sum()) / count, 0)
                doc_length = ss.diags(1. / length, 0)
                # max_counts = ss.diags(1. / X.max(axis=1).A.ravel(), 0)
                X = doc_length * X * bg_rate
                X.data = np.clip(X.data, 0, 1)  # np.log(X.data) / (np.log(X.data) + 1)
        return X

    def initialize_parameters(self, X, words, docs):
        """Store some statistics about X for future use, and initialize alpha, tc"""
        self.n_samples, self.n_visible = X.shape
        if self.n_hidden > 1:
            self.alpha = np.random.random((self.n_hidden, self.n_visible))
            # self.alpha /= np.sum(self.alpha, axis=0, keepdims=True)
        else:
            self.alpha = np.ones((self.n_hidden, self.n_visible), dtype=float)
        self.tc_history = []
        self.tcs = np.zeros(self.n_hidden)
        self.word_counts = np.array(
            X.sum(axis=0)).ravel()  # 1-d array of total word occurrences. (Probably slow for CSR)
        if np.any(self.word_counts == 0) or np.any(self.word_counts == self.n_samples):
            print('WARNING: Some words never appear (or always appear)')
            self.word_counts = self.word_counts.clip(0.01, self.n_samples - 0.01)
        self.word_freq = (self.word_counts).astype(float) / self.n_samples
        self.px_frac = (np.log1p(-self.word_freq) - np.log(self.word_freq)).reshape((-1, 1))  # nv by 1
        self.lp0 = np.log1p(-self.word_freq).reshape((-1, 1))  # log p(x_i=0)
        self.h_x = binary_entropy(self.word_freq)
        if self.verbose:
            print('word counts', self.word_counts)
        # Set column labels
        self.words = words
        if words is not None:
            if len(words) != X.shape[1]:
                print('WARNING: number of column labels != number of columns of X. Check len(words) and X.shape[1]')
            col_index2word = {index:word for index,word in enumerate(words)}
            word2col_index = {word:index for index,word in enumerate(words)}
            self.col_index2word = col_index2word
            self.word2col_index = word2col_index
        else:
            self.col_index2word = None
            self.word2col_index = None
        # Set row labels
        self.docs = docs
        if docs is not None:
            if len(docs) != X.shape[0]:
                print('WARNING: number of row labels != number of rows of X. Check len(docs) and X.shape[0]')
            row_index2doc = {index:doc for index,doc in enumerate(docs)}
            self.row_index2doc = row_index2doc
        else:
            self.row_index2doc = None

    def update_word_parameters(self, X, words):
        """
        updates parameters that need to be changed for each new model update
        specifically, this re-calculates word count related parameters to be based on X,
        where X is a batch of new data
        """
        self.n_samples, self.n_visible = X.shape
        self.word_counts = np.array(
            X.sum(axis=0)).ravel()  # 1-d array of total word occurrences. (Probably slow for CSR)
        if np.any(self.word_counts == 0) or np.any(self.word_counts == self.n_samples):
            print('WARNING: Some words never appear (or always appear)')
            self.word_counts = self.word_counts.clip(0.01, self.n_samples - 0.01)
        self.word_freq = (self.word_counts).astype(float) / self.n_samples
        self.px_frac = (np.log1p(-self.word_freq) - np.log(self.word_freq)).reshape((-1, 1))  # nv by 1
        self.lp0 = np.log1p(-self.word_freq).reshape((-1, 1))  # log p(x_i=0)
        self.h_x = binary_entropy(self.word_freq)
        if self.verbose:
            print('word counts', self.word_counts)
        self.words = words
        if words is not None:
            if len(words) != X.shape[1]:
                print('WARNING: number of column labels != number of columns of X. Check len(words) and X.shape[1]')
            col_index2word = {index:word for index,word in enumerate(words)}
            word2col_index = {word:index for index,word in enumerate(words)}
            self.col_index2word = col_index2word
            self.word2col_index = word2col_index
        else:
            self.col_index2word = None
            self.word2col_index = None

    def preprocess_anchors(self, anchors):
        """Preprocess anchors so that it is a list of column indices if not already"""
        if anchors is not None:
            for n, anchor_list in enumerate(anchors):
                # Check if list of anchors or a single str or int anchor
                if type(anchor_list) is not list:
                    anchor_list = [anchor_list]
                # Convert list of anchors to list of anchor indices
                new_anchor_list = []
                for anchor in anchor_list:
                    # Turn string anchors into index anchors
                    if isinstance(anchor, string_types):
                        if self.words is not None:
                            if anchor in self.word2col_index:
                                new_anchor_list.append(self.word2col_index[anchor])
                            else:
                                raise KeyError('Anchor word not in word column labels provided to CorEx: {}'.format(anchor))
                        else:
                                raise NameError("Provided non-index anchors to CorEx without also providing 'words'")
                    else:
                        new_anchor_list.append(anchor)
                # Update anchors with new anchor list
                if len(new_anchor_list) == 1:
                    anchors[n] = new_anchor_list[0]
                else:
                    anchors[n] = new_anchor_list

        return anchors

    def calculate_p_y(self, p_y_given_x):
        """Estimate log p(y_j=1)."""
        return np.log(np.mean(p_y_given_x, axis=0))  # n_hidden, log p(y_j=1)

    def calculate_theta(self, X, p_y_given_x, log_p_y):
        """Estimate marginal parameters from data and expected latent labels."""
        # log p(x_i=1|y)
        n_samples = X.shape[0]
        p_dot_y = X.T.dot(p_y_given_x).clip(0.01 * np.exp(log_p_y), (n_samples - 0.01) * np.exp(
            log_p_y))  # nv by ns dot ns by m -> nv by m  # TODO: Change to CSC for speed?
        lp_1g1 = np.log(p_dot_y) - np.log(n_samples) - log_p_y
        lp_1g0 = np.log(self.word_counts[:, np.newaxis] - p_dot_y) - np.log(n_samples) - log_1mp(log_p_y)
        lp_0g0 = log_1mp(lp_1g0)
        lp_0g1 = log_1mp(lp_1g1)
        return np.array([lp_0g0, lp_0g1, lp_1g0, lp_1g1])  # 4 by nv by m

    def calculate_alpha(self, X, p_y_given_x, theta, log_p_y, tcs):
        """A rule for non-tree CorEx structure."""
        # TODO: Could make it sparse also? Well, maybe not... at the beginning it's quite non-sparse
        mis = self.calculate_mis(theta, log_p_y)
        if self.n_hidden == 1:
            alphaopt = np.ones((1, self.n_visible))
        elif self.tree:
            # sa = np.sum(self.alpha, axis=0)
            tc_oom = 1. / self.n_samples
            sa = np.sum(self.alpha[tcs > tc_oom], axis=0)
            self.t = np.where(sa > 1.1, 1.3 * self.t, self.t)
            # tc_oom = np.median(self.h_x)  # \propto TC of a small group of corr. variables w/median entropy...
            # t = 20 + (20 * np.abs(tcs) / tc_oom).reshape((self.n_hidden, 1))  # worked well in many tests
            t = (1 + self.t * np.abs(tcs).reshape((self.n_hidden, 1)))
            maxmis = np.max(mis, axis=0)
            for i in np.where((mis == maxmis).sum(axis=0))[0]:  # Break ties for the largest MI
                mis[:, i] += 1e-10 * np.random.random(self.n_hidden)
                maxmis[i] = np.max(mis[:, i])
            with np.errstate(under='ignore'):
                alphaopt = np.exp(t * (mis - maxmis) / self.h_x)
        else:
            # TODO: Can we make a fast non-tree version of update in the AISTATS paper?
            alphaopt = np.zeros((self.n_hidden, self.n_visible))
            top_ys = np.argsort(-mis, axis=0)[:self.tree]
            raise NotImplementedError
        self.mis = mis  # So we don't have to recalculate it when used later
        return alphaopt

    def calculate_latent(self, X, theta):
        """"Calculate the probability distribution for hidden factors for each sample."""
        ns, nv = X.shape
        log_pygx_unnorm = np.empty((2, ns, self.n_hidden))
        c0 = np.einsum('ji,ij->j', self.alpha, theta[0] - self.lp0)
        c1 = np.einsum('ji,ij->j', self.alpha, theta[1] - self.lp0)  # length n_hidden
        info0 = np.einsum('ji,ij->ij', self.alpha, theta[2] - theta[0] + self.px_frac)
        info1 = np.einsum('ji,ij->ij', self.alpha, theta[3] - theta[1] + self.px_frac)
        log_pygx_unnorm[1] = self.log_p_y + c1 + X.dot(info1)
        log_pygx_unnorm[0] = log_1mp(self.log_p_y) + c0 + X.dot(info0)
        return self.normalize_latent(log_pygx_unnorm)

    def normalize_latent(self, log_pygx_unnorm):
        """Normalize the latent variable distribution

        For each sample in the training set, we estimate a probability distribution
        over y_j, each hidden factor. Here we normalize it. (Eq. 7 in paper.)
        This normalization factor is used for estimating TC.

        Parameters
        ----------
        Unnormalized distribution of hidden factors for each training sample.

        Returns
        -------
        p_y_given_x : 3D array, shape (n_hidden, n_samples)
            p(y_j|x^l), the probability distribution over all hidden factors,
            for data samples l = 1...n_samples
        log_z : 2D array, shape (n_hidden, n_samples)
            Point-wise estimate of total correlation explained by each Y_j for each sample,
            used to estimate overall total correlation.

        """
        with np.errstate(under='ignore'):
            log_z = logsumexp(log_pygx_unnorm, axis=0)  # Essential to maintain precision.
            log_pygx = log_pygx_unnorm[1] - log_z
            p_norm = np.exp(log_pygx)
        return p_norm.clip(1e-6, 1 - 1e-6), log_pygx, log_z  # ns by m

    def update_tc(self, log_z):
        self.tcs = np.mean(log_z, axis=0)
        self.tc_history.append(np.sum(self.tcs))

    def print_verbose(self):
        if self.verbose:
            print(self.tcs)
        if self.verbose > 1:
            print(self.alpha[:, :, 0])
            print(self.theta)

    def convergence(self):
        if len(self.tc_history) > 10:
            dist = -np.mean(self.tc_history[-10:-5]) + np.mean(self.tc_history[-5:])
            return np.abs(dist) < self.eps  # Check for convergence.
        else:
            return False

    def __getstate__(self):
        # In principle, if there were variables that are themselves classes... we have to handle it to pickle correctly
        # But I think I programmed around all that.
        self_dict = self.__dict__.copy()
        return self_dict

    def save(self, filename):
        """ Pickle a class instance. E.g., corex.save('saved.dat') """
        # Avoid saving words with object.
        #TODO: figure out why Unicode sometimes causes an issue with loading after pickling
        if self.words is not None:
            temp_words = self.words
            self.words = None
        else:
            temp_words = None
        # Save CorEx object
        import pickle
        if path.dirname(filename) and not path.exists(path.dirname(filename)):
            makedirs(path.dirname(filename))
        pickle.dump(self, open(filename, 'wb'), protocol=-1)
        # Restore words to CorEx object
        self.words = temp_words

    def save_joblib(self, filename):
        """ Serialize a class instance with joblib - better for larger models. E.g., corex.save('saved.dat') """
        # Avoid saving words with object.
        if self.words is not None:
            temp_words = self.words
            self.words = None
        else:
            temp_words = None
        # Save CorEx object
        if path.dirname(filename) and not path.exists(path.dirname(filename)):
            makedirs(path.dirname(filename))
        joblib.dump(self, filename)
        # Restore words to CorEx object
        self.words = temp_words

    def sort_and_output(self, X):
        order = np.argsort(self.tcs)[::-1]  # Order components from strongest TC to weakest
        self.tcs = self.tcs[order]  # TC for each component
        self.alpha = self.alpha[order]  # Connections between X_i and Y_j
        self.log_p_y = self.log_p_y[order]  # Parameters defining the representation
        self.theta = self.theta[:, :, order]  # Parameters defining the representation

    def calculate_mis(self, theta, log_p_y):
        """Return MI in nats, size n_hidden by n_variables"""
        p_y = np.exp(log_p_y).reshape((-1, 1))  # size n_hidden, 1
        mis = self.h_x - p_y * binary_entropy(np.exp(theta[3]).T) - (1 - p_y) * binary_entropy(np.exp(theta[2]).T)
        return (mis - 1. / (2. * self.n_samples)).clip(0.)  # P-T bias correction

    def get_topics(self, n_words=10, topic=None, print_words=True):
        """
        Return list of lists of tuples. Each list consists of the top words for a topic
        and each tuple is a pair (word, mutual information). If 'words' was not provided
        to CorEx, then 'word' will be an integer column index of X

        topic_n : integer specifying which topic to get (0-indexed)
        print_words : boolean, get_topics will attempt to print topics using
                      provided column labels (through 'words') if possible. Otherwise,
                      topics will be consist of column indices of X
        """
        # Determine which topics to iterate over
        if topic is not None:
            topic_ns = [topic]
        else:
            topic_ns = list(range(self.labels.shape[1]))
        # Determine whether to return column word labels or indices
        if self.words is None:
            print_words = False
            print("NOTE: 'words' not provided to CorEx. Returning topics as lists of column indices")
        elif len(self.words) != self.alpha.shape[1]:
            print_words = False
            print('WARNING: number of column labels != number of columns of X. Cannot reliably add labels to topics. Check len(words) and X.shape[1]. Use .set_words() to fix')

        topics = [] # TODO: make this faster, it's slower than it should be
        for n in topic_ns:
            # Get indices of which words belong to the topic
            inds = np.where(self.alpha[n] >= 1.)[0]
            # Sort topic words according to mutual information
            inds = inds[np.argsort(-self.alpha[n,inds] * self.mis[n,inds])]
            # Create topic to return
            if print_words is True:
                topic = [(self.col_index2word[ind], self.sign[n,ind]*self.mis[n,ind]) for ind in inds[:n_words]]
            else:
                topic = [(ind, self.sign[n,ind]*self.mis[n,ind]) for ind in inds[:n_words]]
            # Add topic to list of topics if returning all topics. Otherwise, return topic
            if len(topic_ns) != 1:
                topics.append(topic)
            else:
                return topic

        return topics

    def get_top_docs(self, n_docs=10, topic=None, sort_by='log_prob', print_docs=True):
        """
        Return list of lists of tuples. Each list consists of the top docs for a topic
        and each tuple is a pair (doc, pointwise TC or probability). If 'docs' was not
        provided to CorEx, then each doc will be an integer row index of X

        topic_n : integer specifying which topic to get (0-indexed)
        sort_by: 'log_prob' or 'tc', use either 'log_p_y_given_x' or 'log_z' respectively
                 to return top docs per each topic
        print_docs : boolean, get_top_docs will attempt to print topics using
                     provided row labels (through 'docs') if possible. Otherwise,
                     top docs will be consist of row indices of X
        """
        # Determine which topics to iterate over
        if topic is not None:
            topic_ns = [topic]
        else:
            topic_ns = list(range(self.labels.shape[1]))
        # Determine whether to return row doc labels or indices
        if self.docs is None:
            print_docs = False
            print("NOTE: 'docs' not provided to CorEx. Returning top docs as lists of row indices")
        elif len(self.docs) != self.labels.shape[0]:
            print_words = False
            print('WARNING: number of row labels != number of rows of X. Cannot reliably add labels. Check len(docs) and X.shape[0]. Use .set_docs() to fix')
        # Get appropriate matrix to sort
        if sort_by == 'log_prob':
            doc_values = self.log_p_y_given_x
        elif sort_by == 'tc':
            print('WARNING: sorting by logz not well tested')
            doc_values = self.log_z
        else:
            print("Invalid 'sort_by' parameter, must be 'prob' or 'tc'")
            return
        # Get top docs for each topic
        doc_inds = np.argsort(-doc_values, axis=0)
        top_docs = [] # TODO: make this faster, it's slower than it should be
        for n in topic_ns:
            if print_docs is True:
                topic_docs = [(self.row_index2doc[ind], doc_values[ind,n]) for ind in doc_inds[:n_docs,n]]
            else:
                topic_docs = [(ind, doc_values[ind,n]) for ind in doc_inds[:n_docs,n]]
            # Add docs to list of top docs per topic if returning all topics. Otherwise, return
            if len(topic_ns) != 1:
                top_docs.append(topic_docs)
            else:
                return topic_docs

        return top_docs

    def set_words(self, words):
        self.words = words
        if words is not None:
            if len(words) != self.alpha.shape[1]:
                print('WARNING: number of column labels != number of columns of X. Check len(words) and .alpha.shape[1]')
            col_index2word = {index:word for index,word in enumerate(words)}
            word2col_index = {word:index for index,word in enumerate(words)}
            self.col_index2word = col_index2word
            self.word2col_index = word2col_index

    def set_docs(self, docs):
        self.docs = docs
        if docs is not None:
            if len(docs) != self.labels.shape[0]:
                print('WARNING: number of row labels != number of rows of X. Check len(docs) and .labels.shape[0]')
            row_index2doc = {index:doc for index,doc in enumerate(docs)}
            self.row_index2doc = row_index2doc


def log_1mp(x):
    return np.log1p(-np.exp(x))


def binary_entropy(p):
    return np.where(p > 0, - p * np.log2(p) - (1 - p) * np.log2(1 - p), 0)


def flatten(a):
    b = []
    for ai in a:
        if type(ai) is list:
            b += ai
        else:
            b.append(ai)
    return b


def load(filename):
    """ Unpickle class instance. """
    import pickle
    return pickle.load(open(filename, 'rb'))


def load_joblib(filename):
    """ Load class instance with joblib. """
    return joblib.load(filename)
