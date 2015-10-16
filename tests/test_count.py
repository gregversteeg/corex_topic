# Test the ability to recover clusters of weakly correlated count data
import sys
sys.path.append('..')
import corex_topic as ct
import numpy as np
import scipy.sparse as ss

verbose = True
np.set_printoptions(precision=3, suppress=True, linewidth=200)
seed = 1
np.random.seed(seed)

# n_groups groups with a few variables each, various weak correlations based on word counts
n_samples = 20000
n_groups = 50
doc_length = np.random.choice([300], n_samples)

def observed(s):
    # Generate a randomly sized group of variables weakly correlated to s
    n = np.random.randint(20, 40)

    output = []
    for i in range(n):
        background = np.random.choice([0.005, 0.0005], p=[0.1, 0.9])  # BG Word rates, 1 in 1000 or in 10k
        lam = np.random.choice([2 * background, 3 * background])  # Elevated rates
        output.append(np.random.poisson(doc_length * (s * lam + background)))  # ~100 word doc.
    return np.vstack(output).T


def score(true, predicted):
    """Compare n true signals to some number of predicted signals.
    For each true signal take the min RMSE of each predicted.
    Signals are standardized first."""
    rs = []
    for t in true.T:
        r1 = max(np.mean(t==p) for p in predicted.T)
        r2 = max(np.mean(t==(1-p)) for p in predicted.T)
        rs.append(max(r1, r2))
    return np.array(rs)


baseline = (np.random.random((n_samples, n_groups)) < 0.05)
signal = (np.random.random((n_samples, n_groups)) < 0.05)  # Sparse signals... only 5% ones
data_groups = sorted([observed(s) for s in signal.T], key=lambda q: -q.shape[1])
data = ss.csr_matrix(np.hstack(data_groups))
print 'group sizes', map(lambda q: q.shape[1], data_groups)
print 'Data size:', data.shape, data.nnz
print 'Range of counts', np.min(data.data), np.max(data.data)
print 'Av. doc word count', np.mean(data.sum(axis=1).A.ravel())
print 'Perfect score:', score(signal, signal)
threshold = np.mean(score(signal, baseline))
print 'Baseline score:', threshold

print 'sparse corex, bbow'
out = ct.Corex(n_hidden=n_groups, seed=seed, verbose=verbose, max_iter=100).fit(data>0)
scores = score(signal, out.labels)
print 'TC:', out.tc
print 'Actual score:', scores
print 'Number Ok, %d / %d' % (np.sum(scores > 0.5 * (threshold + 1)), len(scores))
count_score = np.sum(scores > 0.75)

print 'sparse corex, freq'
out = ct.Corex(n_hidden=n_groups, seed=seed, verbose=verbose, max_iter=100).fit(data)
scores = score(signal, out.labels)
print 'TC:', out.tc
print 'Actual score:', scores
print 'Number Ok, %d / %d' % (np.sum(scores > 0.5 * (threshold + 1)), len(scores))
count_score = np.sum(scores > 0.95)

# import corex as ce
# sys.path.append('../../corex_dev')
# print 'traditional corex'
# out = ce.Corex(n_hidden=n_groups, seed=seed, verbose=verbose, max_iter=50, marginal_description='discrete').fit((data>0).A)
# scores = score(signal, out.labels)
# print 'TC:', out.tc
# print 'Actual score:', scores
# print 'Number Ok, %d / %d' % (np.sum(scores > 0.75), len(scores))
# count_score = np.sum(scores > 0.75)