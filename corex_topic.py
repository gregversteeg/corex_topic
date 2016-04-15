"""CorEx Hierarchical Topic Models

Use the principle of Total Cor-relation Explanation (CorEx) to construct hierarchical topic models.
This module is specially designed for sparse count data.

Greg Ver Steeg and Aram Galstyan. "Maximally Informative
Hierarchical Representations of High-Dimensional Data"
AISTATS, 2015. arXiv preprint arXiv:1410.7404.

Code below written by:
Greg Ver Steeg (gregv@isi.edu), 2015.

License: Apache V2
"""

import numpy as np  # Tested with 1.8.0
from os import makedirs
from os import path
from scipy.misc import logsumexp  # Tested with 0.13.0
import scipy.sparse as ss


class Corex(object):
    """
    CorEx hierarchical topic models
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
            print 'corex, rep size:', n_hidden
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

    def fit(self, X, anchors=None, anchor_strength=1):
        """Fit CorEx on the data X. See fit_transform.
        """
        self.fit_transform(X, anchors=anchors, anchor_strength=anchor_strength)
        return self

    def fit_transform(self, X, anchors=None, anchor_strength=1):
        """Fit CorEx on the data

        Parameters
        ----------
        X : scipy sparse CSR or a numpy matrix, shape = [n_samples, n_visible]
            Count data or some other sparse binary data.

        anchors : A list of variables anchor each corresponding latent factor to.

        anchor_strength : How strongly to weight the anchors.

        Returns
        -------
        Y: array-like, shape = [n_samples, n_hidden]
           Learned values for each latent factor for each sample.
           Y's are sorted so that Y_1 explains most correlation, etc.
        """
        X = self.preprocess(X)
        self.initialize_parameters(X)
        p_y_given_x = np.random.random((self.n_samples, self.n_hidden))

        for nloop in range(self.max_iter):
            if nloop > 0:
                for j in np.where(((self.alpha >= 1.) * self.sign).sum(axis=1) < 0)[0]:
                    # Switch Y labels so that p(Y) <= 0.5
                    p_y_given_x[:, j] = 1. - p_y_given_x[:, j]
            self.log_p_y = self.calculate_p_y(p_y_given_x)
            self.theta = self.calculate_theta(X, p_y_given_x, self.log_p_y)  # log p(x_i=1|y)  nv by m by k

            if self.n_hidden > 1 and nloop > 0:  # Structure learning step
                self.alpha = self.calculate_alpha(X, p_y_given_x, self.theta, self.log_p_y, self.tcs)
            if anchors is not None:
##### BEGIN DCK #####
                for A in anchors:
                    try:
                        for a in A:
                            self.alpha[:, a] = 0
                    except:
                        self.alpha[:, A] = 0
                for ia, A in enumerate(anchors):
                    try:
                        for a in A:
                            self.alpha[ia, a] = anchor_strength
                    except:
                        self.alpha[ia, A] = anchor_strength
##### END DCK #####
            p_y_given_x, log_z = self.calculate_latent(X, self.theta)

            self.update_tc(log_z)  # Calculate TC and record history to check convergence
            self.print_verbose()
            if self.convergence():
                break

        if self.verbose:
            print 'Overall tc:', self.tc

        if anchors is None:
            self.sort_and_output(X)
        self.p_y_given_x, self.log_z = self.calculate_latent(X, self.theta)  # Needed to output labels
        self.mis = self.calculate_mis(self.theta, self.log_p_y)  # / self.h_x  # could normalize MIs
        return self.labels


    def transform(self, X, details=False):
        """
        Label hidden factors for (possibly previously unseen) samples of data.
        Parameters: samples of data, X, shape = [n_samples, n_visible]
        Returns: , shape = [n_samples, n_hidden]
        """
        X = self.preprocess(X)
        p_y_given_x, log_z = self.calculate_latent(X, self.theta)
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

    def initialize_parameters(self, X):
        """Store some statistics about X for future use, and initialize alpha, tc"""
        self.n_samples, self.n_visible = X.shape
        if self.n_hidden > 1:
            self.alpha = np.random.random((self.n_hidden, self.n_visible))
            # self.alpha /= np.sum(self.alpha, axis=0, keepdims=True)
        else:
            self.alpha = np.ones((self.n_hidden, self.n_visible), dtype=float)
        self.tc_history = []
        self.tcs = np.zeros(self.n_hidden)
        self.word_counts = np.array(X.sum(axis=0)).ravel()  # 1-d array of total word occurrences. (Probably slow for CSR)
        if np.any(self.word_counts == 0) or np.any(self.word_counts == self.n_samples):
            print 'warning: Some words never appear (or always appear)'
            self.word_counts = self.word_counts.clip(0.01, self.n_samples - 0.01)
        self.word_freq = (self.word_counts).astype(float) / self.n_samples
        self.px_frac = (np.log1p(-self.word_freq) - np.log(self.word_freq)).reshape((-1, 1))  # nv by 1
        self.lp0 = np.log1p(-self.word_freq).reshape((-1, 1))  # log p(x_i=0)
        self.h_x = binary_entropy(self.word_freq)
        if self.verbose:
            print 'word counts', self.word_counts

    def calculate_p_y(self, p_y_given_x):
        """Estimate log p(y_j=1)."""
        return np.log(np.mean(p_y_given_x, axis=0))  # n_hidden, log p(y_j=1)

    def calculate_theta(self, X, p_y_given_x, log_p_y):
        """Estimate marginal parameters from data and expected latent labels."""
        # log p(x_i=1|y)
        n_samples = X.shape[0]
        p_dot_y = X.T.dot(p_y_given_x).clip(0.01 * np.exp(log_p_y), (n_samples - 0.01) * np.exp(log_p_y))  # nv by ns dot ns by m -> nv by m  # TODO: Change to CSC for speed?
        lp_1g1 = np.log(p_dot_y) - np.log(n_samples) - log_p_y
        lp_1g0 = np.log(self.word_counts[:, np.newaxis] - p_dot_y) - np.log(n_samples) - log_1mp(self.log_p_y)
        lp_0g0 = log_1mp(lp_1g0)
        lp_0g1 = log_1mp(lp_1g1)
        return np.array([lp_0g0, lp_0g1, lp_1g0, lp_1g1])  # 4 by nv by m

    def calculate_alpha(self, X, p_y_given_x, theta, log_p_y, tcs):
        """A rule for non-tree CorEx structure."""
        # TODO: Could make it sparse also? Well, maybe not... at the beginning it's quite non-sparse
        if self.tree:
            # sa = np.sum(self.alpha, axis=0)
            tc_oom = 1. / self.n_samples
            sa = np.sum(self.alpha[tcs > tc_oom], axis=0)
            self.t = np.where(sa > 1.1, 1.3 * self.t, self.t)
            mis = self.calculate_mis(theta, log_p_y)
            #tc_oom = np.median(self.h_x)  # \propto TC of a small group of corr. variables w/median entropy...
            #t = 20 + (20 * np.abs(tcs) / tc_oom).reshape((self.n_hidden, 1))  # worked well in many tests
            t = (1 + self.t * np.abs(tcs).reshape((self.n_hidden, 1)))
            maxmis = np.max(mis, axis=0)
            with np.errstate(under='ignore'):
                alphaopt = np.exp(t * (mis - maxmis) / self.h_x)
            return alphaopt
        else:
            # TODO: Can we make a fast non-tree version of update in the AISTATS paper?
            alpha = np.zeros((self.n_hidden, self.n_visible))
            mis = self.calculate_mis(theta, log_p_y)
            top_ys = np.argsort(-mis, axis=0)[:self.tree]
            raise NotImplementedError
        return alpha

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
        pygx, log_z = self.normalize_latent(log_pygx_unnorm)
        return pygx, log_z

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
            p_norm = np.exp(log_pygx_unnorm[1] - log_z)
        return p_norm.clip(1e-6, 1-1e-6), log_z  # ns by m

    def update_tc(self, log_z):
        self.tcs = np.mean(log_z, axis=0)
        self.tc_history.append(np.sum(self.tcs))

    def print_verbose(self):
        if self.verbose:
            print self.tcs
        if self.verbose > 1:
            print self.alpha[:, :, 0]
            print self.theta

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
        import pickle
        if path.dirname(filename) and not path.exists(path.dirname(filename)):
            makedirs(path.dirname(filename))
        pickle.dump(self, open(filename, 'w'), protocol=-1)

    def load(self, filename):
        """ Unpickle class instance. E.g., corex = ce.Marginal_Corex().load('saved.dat') """
        import pickle
        return pickle.load(open(filename))

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


def log_1mp(x):
    return np.log1p(-np.exp(x))


def binary_entropy(p):
    return np.where(p > 0, - p * np.log2(p) - (1 - p) * np.log2(1 - p), 0)
