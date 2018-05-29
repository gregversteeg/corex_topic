""" This module implements some visualizations based on CorEx representations.
"""

import os
from shutil import copyfile
import codecs
import numpy as np
import matplotlib
matplotlib.use('Agg') # to create visualizations on a display-less server
import pylab
import networkx as nx
import textwrap
import scipy.sparse as ss
import sklearn.feature_extraction.text as skt
#import cPickle, pickle # neither module is used, and cPickle is not part of Anaconda build, so commented for LF run
import corextopic as ct
import sys, traceback
from time import time
import re
import sklearn.feature_extraction.text as skt
from nltk.stem.snowball import *
pattern = '\\b[A-Za-z]+\\b'
np.seterr(all='ignore')


def vis_rep(corex, data=None, row_label=None, column_label=None, prefix='topics'):
    """Various visualizations and summary statistics for a one layer representation"""
    if column_label is None:
        column_label = list(map(str, range(data.shape[1])))
    if row_label is None:
        row_label = list(map(str, range(corex.n_samples)))

    alpha = corex.alpha

    print('Print topics in text file')
    output_groups(corex.tcs, alpha, corex.mis, column_label, corex.sign, prefix=prefix)
    output_labels(corex.labels, row_label, prefix=prefix)
    output_cont_labels(corex.p_y_given_x, row_label, prefix=prefix)
    output_strong(corex.tcs, alpha, corex.mis, corex.labels, prefix=prefix)
    anomalies(corex.log_z, row_label=row_label, prefix=prefix)
    plot_convergence(corex.tc_history, prefix=prefix)
    if data is not None:
        plot_heatmaps(data, alpha, corex.mis, column_label, corex.p_y_given_x, prefix=prefix)


def vis_hierarchy(corexes, column_label=None, max_edges=100, prefix='topics', n_anchors=0):
    """Visualize a hierarchy of representations."""
    if column_label is None:
        column_label = list(map(str, range(corexes[0].alpha.shape[1])))

    # make l1 label
    alpha = corexes[0].alpha
    mis = corexes[0].mis
    l1_labels = []
    annotate = lambda q, s: q if s > 0 else '~' + q
    for j in range(corexes[0].n_hidden):
        # inds = np.where(alpha[j] * mis[j] > 0)[0]
        inds = np.where(alpha[j] >= 1.)[0]
        inds = inds[np.argsort(-alpha[j, inds] * mis[j, inds])]
        group_number = u"red_" + unicode(j) if j < n_anchors else unicode(j)
        label = group_number + u':' + u' '.join([annotate(column_label[ind], corexes[0].sign[j,ind]) for ind in inds[:6]])
        label = textwrap.fill(label, width=25)
        l1_labels.append(label)

    # Construct non-tree graph
    weights = [corex.alpha.clip(0, 1) * corex.mis for corex in corexes[1:]]
    node_weights = [corex.tcs for corex in corexes[1:]]
    g = make_graph(weights, node_weights, l1_labels, max_edges=max_edges)

    # Display pruned version
    h = g.copy()  # trim(g.copy(), max_parents=max_parents, max_children=max_children)
    edge2pdf(h, prefix + '/graphs/graph_prune_' + str(max_edges), labels='label', directed=True, makepdf=True)

    # Display tree version
    tree = g.copy()
    tree = trim(tree, max_parents=1, max_children=False)
    edge2pdf(tree, prefix + '/graphs/tree', labels='label', directed=True, makepdf=True)

    # Output JSON files
    try:
        import os
        copyfile(os.path.dirname(os.path.realpath(__file__)) + '/tests/d3_files/force.html', prefix + '/graphs/force.html')
    except:
        print("Couldn't find 'force.html' file for visualizing d3 output")
    import json
    from networkx.readwrite import json_graph

    mapping = dict([(n, tree.node[n].get('label', str(n))) for n in tree.nodes()])
    tree = nx.relabel_nodes(tree, mapping)
    json.dump(json_graph.node_link_data(tree), safe_open(prefix + '/graphs/force.json', 'w+'))
    json.dump(json_graph.node_link_data(h), safe_open(prefix + '/graphs/force_nontree.json', 'w+'))

    return g


def plot_heatmaps(data, alpha, mis, column_label, cont, topk=40, athresh=0.2, prefix=''):
    import seaborn as sns
    cmap = sns.cubehelix_palette(as_cmap=True, light=.9)
    import matplotlib.pyplot as plt
    m, nv = mis.shape
    for j in range(m):
        inds = np.where(np.logical_and(alpha[j] > athresh, mis[j] > 0.))[0]
        inds = inds[np.argsort(- alpha[j, inds] * mis[j, inds])][:topk]
        if len(inds) >= 2:
            plt.clf()
            order = np.argsort(cont[:,j])
            if type(data) == np.ndarray:
                subdata = data[:, inds][order].T
            else:
                # assume sparse
                subdata = data[:, inds].toarray()
                subdata = subdata[order].T
            columns = [column_label[i] for i in inds]
            fig, ax = plt.subplots(figsize=(20, 10))
            sns.heatmap(subdata, vmin=0, vmax=1, cmap=cmap, yticklabels=columns, xticklabels=False, ax=ax, cbar_kws={"ticks": [0, 0.5, 1]})
            plt.yticks(rotation=0)
            filename = '{}/heatmaps/group_num={}.png'.format(prefix, j)
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            plt.title("Latent factor {}".format(j))
            plt.savefig(filename, bbox_inches='tight')
            plt.close('all')
            #plot_rels(data[:, inds], map(lambda q: column_label[q], inds), colors=cont[:, j],
            #          outfile=prefix + '/relationships/group_num=' + str(j), latent=labels[:, j], alpha=0.1)


def make_graph(weights, node_weights, column_label, max_edges=100):
    all_edges = np.hstack(list(map(np.ravel, weights)))
    max_edges = min(max_edges, len(all_edges))
    w_thresh = np.sort(all_edges)[-max_edges]
    print('weight threshold is %f for graph with max of %f edges ' % (w_thresh, max_edges))
    g = nx.DiGraph()
    max_node_weight = max([max(w) for w in node_weights])
    for layer, weight in enumerate(weights):
        m, n = weight.shape
        for j in range(m):
            g.add_node((layer + 1, j))
            g.node[(layer + 1, j)]['weight'] = 0.3 * node_weights[layer][j] / max_node_weight
            for i in range(n):
                if weight[j, i] > w_thresh:
                    if weight[j, i] > w_thresh / 2:
                        g.add_weighted_edges_from([( (layer, i), (layer + 1, j), 10 * weight[j, i])])
                    else:
                        g.add_weighted_edges_from([( (layer, i), (layer + 1, j), 0)])

    # Label layer 0
    for i, lab in enumerate(column_label):
        g.add_node((0, i))
        g.node[(0, i)]['label'] = lab
        g.node[(0, i)]['name'] = lab  # JSON uses this field
        g.node[(0, i)]['weight'] = 1
    return g


def trim(g, max_parents=False, max_children=False):
    for node in g:
        if max_parents:
            parents = list(g.successors(node))
            weights = [g.edge[node][parent]['weight'] for parent in parents]
            for weak_parent in np.argsort(weights)[:-max_parents]:
                g.remove_edge(node, parents[weak_parent])
        if max_children:
            children = g.predecessors(node)
            weights = [g.edge[child][node]['weight'] for child in children]
            for weak_child in np.argsort(weights)[:-max_children]:
                g.remove_edge(children[weak_child], node)
    return g


def output_groups(tcs, alpha, mis, column_label, direction, thresh=0, prefix=''):
    f = safe_open(prefix + '/groups.txt', 'w+')
    h = safe_open(prefix + '/topics.txt', 'w+')
    m, nv = mis.shape
    annotate = lambda q, s: q if s >= 0 else '~' + q
    for j in range(m):
        f.write('Group num: %d, TC(X;Y_j): %0.3f\n' % (j, tcs[j]))
        # inds = np.where(alpha[j] * mis[j] > thresh)[0]
        inds = np.where(alpha[j] >= 1.)[0]
        inds = inds[np.argsort(-alpha[j, inds] * mis[j, inds])]
        for ind in inds:
            f.write(column_label[ind] + u', %0.3f, %0.3f, %0.3f\n' % (
                mis[j, ind], alpha[j, ind], mis[j, ind] * alpha[j, ind]))
        #h.write(unicode(j) + u':' + u','.join([annotate(column_label[ind], direction[j,ind]) for ind in inds[:10]]) + u'\n')
        h.write(str(j) + u':' + u','.join(
            [annotate(column_label[ind], direction[j, ind]) for ind in inds[:10]]) + u'\n')
    f.close()
    h.close()


def output_labels(labels, row_label, prefix=''):
    f = safe_open(prefix + '/labels.txt', 'w+')
    ns, m = labels.shape
    for l in range(ns):
        f.write(row_label[l] + ',' + ','.join(list(map(lambda q: '%d' % q, labels[l, :])))+ '\n')
    f.close()


def output_cont_labels(p_y_given_x, row_label, prefix=''):
    f = safe_open(prefix + '/cont_labels.txt', 'w+')
    ns, m = p_y_given_x.shape
    for l in range(ns):
        f.write(row_label[l] + ',' + ','.join(list(map(lambda q: '{:.10f}'.format(q), np.log(p_y_given_x[l, :])))) + '\n')
    f.close()


def output_strong(tcs, alpha, mis, labels, prefix=''):
    f = safe_open(prefix + '/most_deterministic_groups.txt', 'w+')
    m, n = alpha.shape
    topk = 5
    ixy = np.clip(np.sum(alpha * mis, axis=1) - tcs, 0, np.inf)
    hys = np.array([entropy(labels[:, j]) for j in range(m)]).clip(1e-6)
    ntcs = [(np.sum(np.sort(alpha[j] * mis[j])[-topk:]) - ixy[j]) / ((topk - 1) * hys[j]) for j in range(m)]

    f.write('Group num., NTC\n')
    for j, ntc in sorted(enumerate(ntcs), key=lambda q: -q[1]):
        f.write('%d, %0.3f\n' % (j, ntc))
    f.close()


def anomalies(log_z, row_label=None, prefix=''):
    from scipy.special import erf

    ns = log_z.shape[0]
    if row_label is None:
        row_label = list(map(str, range(ns)))
    a_score = np.sum(log_z[:, :], axis=1)
    mean, std = np.mean(a_score), np.std(a_score)
    a_score = (a_score - mean) / std
    percentile = 1. / ns
    anomalies = np.where(0.5 * (1 - erf(a_score / np.sqrt(2)) ) < percentile)[0]
    f = safe_open(prefix + '/anomalies.txt', 'w+')
    for i in anomalies:
        f.write(row_label[i] + ', %0.1f\n' % a_score[i])
    f.close()


# Utilities
# IT UTILITIES
def entropy(xsamples):
    # sample entropy for one discrete var
    xsamples = np.asarray(xsamples)
    xsamples = xsamples[xsamples >= 0]  # by def, -1 means missing value
    xs = np.unique(xsamples)
    ns = len(xsamples)
    ps = np.array([float(np.count_nonzero(xsamples == x)) / ns for x in xs])
    return -np.sum(ps * np.log(ps))


def safe_open(filename, mode):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    return codecs.open(filename, mode, "utf-8")


# Visualization utilities

def neato(fname, position=None, directed=False):
    if directed:
        os.system(
            "sfdp " + fname + ".dot -Tpdf -Earrowhead=none -Nfontsize=16  -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True -Gpackmode=clust -Gsep=0.01 -Gsplines=False -o " + fname + "_sfdp.pdf")
        os.system(
            "sfdp " + fname + ".dot -Tpdf -Earrowhead=none -Nfontsize=16  -GK=2 -Gmaxiter=1000 -Goverlap=False -Gpack=True -Gpackmode=clust -Gsep=0.01 -Gsplines=True -o " + fname + "_sfdp_w_splines.pdf")
        return True
    if position is None:
        os.system("neato " + fname + ".dot -Tpdf -o " + fname + ".pdf")
        os.system("fdp " + fname + ".dot -Tpdf -o " + fname + "fdp.pdf")
    else:
        os.system("neato " + fname + ".dot -Tpdf -n -o " + fname + ".pdf")
    return True


def extract_color(label):
    import matplotlib

    colors = matplotlib.colors.cnames.keys()
    parts = label.split('_')
    for part in parts:
        if part in colors:
            parts.remove(part)
            return '_'.join(parts), part
    return label, 'black'


def edge2pdf(g, filename, threshold=0, position=None, labels=None, connected=True, directed=False, makepdf=True):
    #This function will takes list of edges and a filename
    #and write a file in .dot format. Readable, eg. by omnigraffle
    # OR use "neato file.dot -Tpng -n -o file.png"
    # The -n option says whether to use included node positions or to generate new ones
    # for a grid, positions = [(i%28,i/28) for i in range(784)]
    def cnn(node):
        #change node names for dot format
        if type(node) is tuple or type(node) is list:
            return u'n' + u'_'.join(list(map(unicode, node)))
        else:
            return unicode(node)

    if connected:
        touching = list(set(sum([[a, b] for a, b in g.edges()], [])))
        g = nx.subgraph(g, touching)
        print('non-isolated nodes,edges', len(list(g.nodes())), len(list(g.edges())))
    f = safe_open(filename + '.dot', 'w+')
    if directed:
        f.write("strict digraph {\n".encode('utf-8'))
    else:
        f.write("strict graph {\n".encode('utf-8'))
    #f.write("\tgraph [overlap=scale];\n".encode('utf-8'))
    f.write("\tnode [shape=point];\n".encode('utf-8'))
    for a, b, d in g.edges(data=True):
        if d.has_key('weight'):
            if directed:
                f.write(("\t" + cnn(a) + ' -> ' + cnn(b) + ' [penwidth=%.2f' % float(
                    np.clip(d['weight'], 0, 9)) + '];\n').encode('utf-8'))
            else:
                if d['weight'] > threshold:
                    f.write(("\t" + cnn(a) + ' -- ' + cnn(b) + ' [penwidth=' + str(3 * d['weight']) + '];\n').encode(
                        'utf-8'))
        else:
            if directed:
                f.write(("\t" + cnn(a) + ' -> ' + cnn(b) + ';\n').encode('utf-8'))
            else:
                f.write(("\t" + cnn(a) + ' -- ' + cnn(b) + ';\n').encode('utf-8'))
    for n in g.nodes():
        if labels is not None:
            if type(labels) == dict or type(labels) == list:
                thislabel = labels[n].replace(u'"', u'\\"')
                lstring = u'label="' + thislabel + u'",shape=none'
            elif type(labels) == str:
                if g.node[n].has_key('label'):
                    thislabel = g.node[n][labels].replace(u'"', u'\\"')
                    # combine dupes
                    #llist = thislabel.split(',')
                    #thislabel = ','.join([l for l in set(llist)])
                    thislabel, thiscolor = extract_color(thislabel)
                    lstring = u'label="%s",shape=none,fontcolor="%s"' % (thislabel, thiscolor)
                else:
                    weight = g.node[n].get('weight', 0.1)
                    if n[0] == 1:
                        lstring = u'shape=circle,margin="0,0",style=filled,fillcolor=black,fontcolor=white,height=%0.2f,label="%d"' % (
                            2 * weight, n[1])
                    else:
                        lstring = u'shape=point,height=%0.2f' % weight
            else:
                lstring = 'label="' + str(n) + '",shape=none'
            lstring = unicode(lstring)
        else:
            lstring = False
        if position is not None:
            if position == 'grid':
                position = [(i % 28, 28 - i / 28) for i in range(784)]
            posstring = unicode('pos="' + str(position[n][0]) + ',' + str(position[n][1]) + '"')
        else:
            posstring = False
        finalstring = u' [' + u','.join([ts for ts in [posstring, lstring] if ts]) + u']\n'
        #finalstring = u' ['+lstring+u']\n'
        f.write(u'\t' + cnn(n) + finalstring)
    f.write("}".encode('utf-8'))
    f.close()
    if makepdf:
        neato(filename, position=position, directed=directed)
    return True


def predictable(out, data, wdict=None, topk=5, outfile='sorted_groups.txt', graphs=False, prefix='', athresh=0.5,
                tvalue=0.1):
    alpha, labels, lpygx, mis, lasttc = out[:5]
    ns, m = labels.shape
    m, nv = mis.shape
    hys = [entropy(labels[:, j]) for j in range(m)]
    #alpha = np.array([z[2] for z in zs]) # m by nv
    nmis = []
    ixys = []
    for j in range(m):
        if hys[j] > 0:
            #ixy = np.dot((alpha[j]>0.95).astype(int),mis[j])-lasttc[-1][j]
            ixy = max(0., np.dot(alpha[j], mis[j]) - lasttc[-1][j])
            ixys.append(ixy)
            tcn = (np.sum(np.sort(alpha[j] * mis[j])[-topk:]) - ixy) / ((topk - 1) * hys[j])
            nmis.append(tcn)  #ixy) #/hys[j])
        else:
            ixys.append(0)
            nmis.append(0)
    f = safe_open(prefix + outfile, 'w+')
    print(list(enumerate(np.argsort(-np.array(nmis)))))
    print(','.join(list(map(str, list(np.argsort(-np.array(nmis)))))))
    for i, top in enumerate(np.argsort(-np.array(nmis))):
        f.write('Group num: %d, Score: %0.3f\n' % (top, nmis[top]))
        inds = np.where(alpha[top] > athresh)[0]
        inds = inds[np.argsort(-mis[top, inds])]
        for ind in inds:
            f.write(wdict[ind] + ', %0.3f\n' % (mis[top, ind] / np.log(2)))
        if wdict:
            print(','.join(list(map(lambda q: wdict[q], inds))))
            print(','.join(list(map(str, inds))))
        print(top)
        print(nmis[top], ixys[top], hys[top], ixys[top] / hys[top])  #,lasttc[-1][top],hys[top],lasttc[-1][top]/hys[top]
        if graphs:
            print(inds)
            if len(inds) >= 2:
                plot_rels(data[:, inds[:5]], list(map(lambda q: wdict[q], inds[:5])),
                          outfile='relationships/' + str(i) + '_group_num=' + str(top), latent=out[1][:, top],
                          alpha=tvalue)
    f.close()
    return nmis


def shorten(s, n=12):
    if len(s) > 2 * n:
        return s[:n] + '..' + s[-n:]
    return s


def plot_convergence(tc_history, prefix=''):
    pylab.plot(tc_history)
    pylab.xlabel('number of iterations')
    filename = prefix + '/convergence.pdf'
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    pylab.savefig(filename)
    pylab.close('all')
    return True

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


# Utilities to construct generalized binary bag of words matrices


def av_bbow(docs, n=100):
    """Average binary bag of words if we take chunks of a doc of size n"""
    proc = skt.CountVectorizer(token_pattern=pattern)
    proc.fit(docs)
    n_doc, n_words = len(docs), len(proc.vocabulary_)
    mat = ss.lil_matrix((n_doc, n_words))
    for l, doc in enumerate(docs):
        subdocs = chunks(doc, n=n)
        mat[l] = (proc.transform(subdocs) > 0).mean(axis=0).A.ravel()
    return mat.asformat('csr'), proc


def bow(docs):
    """Standard bag of words"""
    proc = skt.CountVectorizer(token_pattern=pattern)
    return proc.fit_transform(docs), proc


def all_bbow(docs, n=100):
    """Split each document into a subdocuments of size n, and return as binary BOW"""
    proc = skt.CountVectorizer(token_pattern=pattern)
    proc.fit(docs)
    ids = []
    for l, doc in enumerate(docs):
        subdocs = chunks(doc, n=n)
        submat = (proc.transform(subdocs) > 0)
        if l == 0:
          mat = submat
        else:
          mat = ss.vstack([mat, submat])
        ids += [l]*submat.shape[0]
    return mat.asformat('csr'), proc, ids


def file_to_array(filename, stemming=False, strategy=2, words_per_doc=100, n_words=10000):
    pattern = '\\b[A-Za-z]+\\b'
    stemmer = SnowballStemmer('english')

    with open(filename, 'rU') as input_file:
        docs = []
        for line in input_file:
            if stemming:
                docs.append(' '.join([stemmer.stem(w) for w in re.findall(pattern, line)]))
            else:
                docs.append(' '.join([w for w in re.findall(pattern, line)]))
    print('processing file')

    if strategy == 1:
        X, proc = av_bbow(docs, n=words_per_doc)
    elif strategy == 2:
        X, proc, ids = all_bbow(docs, n=words_per_doc)
    else:
        X, proc = bow(docs)

    var_order = np.argsort(-X.sum(axis=0).A1)[:n_words]
    X = X[:, var_order]

    #Dictionary
    ivd = {v: k for k, v in proc.vocabulary_.items()}
    words = [ivd[v] for v in var_order]
    return X, words

if __name__ == '__main__':
    # Command line interface
    # Sample commands:
    # python vis_topic.py tests/data/twenty.txt --n_words=2000 --layers=20,3,1 -v --edges=50 -o test_output
    from optparse import OptionParser, OptionGroup


    parser = OptionParser(usage="usage: %prog [options] data_file.csv \n"
                                "Assume one document on each line.")

    group = OptionGroup(parser, "Options")
    group.add_option("-n", "--n_words",
                     action="store", dest="n_words", type="int", default=10000,
                     help="Maximum number of words to include in dictionary.")
    group.add_option("-l", "--layers", dest="layers", type="string", default="2,1",
                     help="Specify number of units at each layer: 5,3,1 has "
                          "5 units at layer 1, 3 at layer 2, and 1 at layer 3")
    group.add_option("-t", "--strategy", dest="strategy", type="int", default=0,
                     help="Specify the strategy for handling non-binary count data.\n"
                          "0. Naive binarization. This will be good for documents of similar length and especially"
                          "short documents.\n"
                          "1. Average binary bag of words. We split documents into chunks, compute the binary "
                          "bag of words for each documents and then average. This implicitly weights all documents"
                          "equally.\n"
                          "2. All binary bag of words. Split documents into chunks and consider each chunk as its"
                          "own binary bag of words documents. This changes the number of documents so it may take"
                          "some work to match the ids back, if desired.\n"
                          "3. Fractional counts. This converts counts into a fraction of the background rate, with 1 as"
                          "the max. Short documents tend to stay binary and words in long documents are weighted"
                          "according to their frequency with respect to background in the corpus.")
    group.add_option("-o", "--output",
                     action="store", dest="output", type="string", default="topic_output",
                     help="A directory to put all output files.")
    group.add_option("-s", "--stemming",
                     action="store_false", dest="stemming", default=True,
                     help="Use a stemmer on words.")
    group.add_option("-v", "--verbose",
                     action="store_true", dest="verbose", default=False,
                     help="Print rich outputs while running.")
    group.add_option("-w", "--words_per_doc",
                     action="store", dest="words_per_doc", type="int", default=300,
                     help="If using all_bbow or av_bbow, this specifies the number of words each "
                          "to split documents into.")
    group.add_option("-e", "--edges",
                     action="store", dest="max_edges", type="int", default=1000,
                     help="Show at most this many edges in graphs.")
    group.add_option("-q", "--regraph",
                     action="store_true", dest="regraph", default=False,
                     help="Don't re-run corex, just re-generate outputs (with number of edges changed).")
    parser.add_option_group(group)

    (options, args) = parser.parse_args()
    if not len(args) == 1:
        print("Run with '-h' option for usage help.")
        sys.exit()

    layers = list(map(int, options.layers.split(',')))
    if layers[-1] != 1:
        layers.append(1)  # Last layer has one unit for convenience so that graph is fully connected.

    #Load data from text file
    print('reading file')
    X, words = file_to_array(args[0], stemming=options.stemming, strategy=options.strategy,
                             words_per_doc=options.words_per_doc, n_words=options.n_words)
    # cPickle.dump(words, open(options.prefix + '/dictionary.dat', 'w'))  # TODO: output dictionary

    # Run CorEx on data
    if options.verbose:
        np.set_printoptions(precision=3, suppress=True)  # For legible output from numpy
        print('\nData summary: X has %d rows and %d columns' % X.shape)
        print('Variable names are: ' + ','.join(words))
        print('Getting CorEx results')
    if options.strategy == 3:
        count = 'fraction'
    else:
        count = 'binarize'  # Strategies 1 and 2 already produce counts <= 1 and are not affected by this choice.
    if not options.regraph:
        for l, layer in enumerate(layers):
            if options.verbose:
                print("Layer ", l)
            if l == 0:
                t0 = time()
                corexes = [ct.Corex(n_hidden=layer, verbose=options.verbose, count=count).fit(X)]
                print('Time for first layer: %0.2f' % (time() - t0))
            else:
                X_prev = np.matrix(corexes[-1].labels)
                corexes.append(ct.Corex(n_hidden=layer, verbose=options.verbose).fit(X_prev))
        for l, corex in enumerate(corexes):
            # The learned model can be loaded again using ct.Corex().load(filename)
            print('TC at layer %d is: %0.3f' % (l, corex.tc))
            corex.save(options.output + '/layer_' + str(l) + '.dat')
    else:
        corexes = [ct.Corex().load(options.output + '/layer_' + str(l) + '.dat') for l in range(len(layers))]


    # This line outputs plots showing relationships at the first layer
    vis_rep(corexes[0], data=X, column_label=words, prefix=options.output)
    # This line outputs a hierarchical networks structure in a .dot file in the "graphs" folder
    # And it tries to compile the dot file into a pdf using the command line utility sfdp (part of graphviz)
    vis_hierarchy(corexes, column_label=words, max_edges=options.max_edges, prefix=options.output)
