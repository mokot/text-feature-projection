# -*- coding: utf-8 -*-

from os import listdir
from os.path import join
import random
from matplotlib import pyplot as plt

import numpy as np
import re
import collections
from unidecode import unidecode

from sklearn.manifold import MDS, TSNE

N = 100
NUM_ITERATIONS = 100 
PLOT_SAVE = False

# color palette for random selection of colors
color_palette = [
    "lightcoral", "firebrick", "coral", "orangered", "peru", "orange",
    "yellowgreen", "greenyellow", "darkseagreen", "palegreen", "springgreen",
    "aquamarine", "turquoise", "teal", "deepskyblue", "royalblue", "lavender",
    "slateblue", "darkorchid", "violet", "hotpink", "gold", "olive", "khaki",
    "lightpink"
]


def triplets(text, n):
    """
    Calculate the triplets of length n in the text.
    """
    # convert to lowercase and remove all non-alphabetic characters
    text = unidecode(text.lower())

    # separate the text into words
    words = [word for word in re.split("[^a-z]", text) if word != ""]
    words = " ".join(words)

    # split the words into n-triplets of length n
    chunks = [words[i:i + n] for i in range(len(words) - (n - 1))]

    # calculate the frequency of each triplet
    n_triplets = dict(collections.Counter(chunks))
    return n_triplets


def read_data(n_triplets):
    """
    Reads the data from the files in the data folder.
    """
    lds = {}
    for fn in listdir("text01"):
        if fn.lower().endswith(".txt"):
            with open(join("text01", fn), encoding="utf8") as f:
                text = f.read()
                nter = triplets(text, n=n_triplets)
                lds[fn] = nter
    return lds


def cosine_dist(d1, d2):
    """
    Calculates the cosine distance between two dictionaries.
    """
    if (not len(d1.keys() & d2.keys())):
        return 1

    temp_data = np.sum(np.array([(d1[key] * d2[key], d1[key]**2, d2[key]**2)
                                 for key in d1.keys() & d2.keys()]),
                       axis=0)
    dot_prod, norm = temp_data[0], temp_data[1]**0.5 * temp_data[2]**0.5

    if not norm:
        return 1

    dist = 1 - (dot_prod / norm)
    return dist if dist > 0 else 0


def distance_mat(data):
    """
    Calculates the cosine distance matrix for the data.
    """
    dist = np.zeros((len(data), len(data)))
    for i, key in enumerate(data):
        dist[i] = np.asarray([
            np.sum(cosine_dist(data[key], data[temp_key])) for temp_key in data
        ])
    return dist


def prepare_data_matrix(data_dict):
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf (NOT the complete tf-idf) measure.
    """
    documents = data_dict.keys()
    X = np.zeros((len(documents), N))

    # find the 100 most common triplets
    data = collections.defaultdict(lambda: 0, {})
    
    # calculate the idf for each triplet
    for doc_key in documents:
        for triple in data_dict[doc_key]:
            data[triple] += 1
    counter = collections.Counter(data)
    data = counter.most_common(N)
    data = np.asarray(data)[:, 0] 

    # for each document, calculate the tf-idf for each triplet
    for i, doc_key in enumerate(documents):
        X[i] = np.asarray([
            data_dict[doc_key][triple_key]
            if triple_key in data_dict[doc_key].keys() else 0
            for triple_key in data
        ])

    return X / np.sum(X), np.asarray(
        [re.split('-|\.', language)[-2] for language in documents])


def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)

    wiki: https://en.wikipedia.org/wiki/Power_iteration
    """

    # calculate covariance matrix
    X = np.cov(X.T)

    # initialize the power iteration with a random vector
    vec = np.random.rand(X.shape[0])

    # repeat the power iteration until the vector remains unchanged
    for _ in range(NUM_ITERATIONS):
        vec_temp = np.dot(X, vec)
        vec_norm = np.linalg.norm(vec_temp)
        vec_old = np.copy(vec)
        vec = np.copy(vec_temp / vec_norm)

        if ((vec == vec_old).all()):
            break

    # calculate the corresponding eigenvalue
    val = vec.T.dot(X).dot(vec)
    return vec, val


def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """
    # calculate eigen vector and corresponding largest eigen value
    vec1, val1 = power_iteration(X)

    # project the data onto the first eigenvector
    data_proj = project_to_eigenvectors(X, np.asarray([vec1]))
    X = np.subtract(X, vec1 * data_proj)

    # calculate eigen vector and corresponding second largest eigen value
    vec2, val2 = power_iteration(X)

    return np.asarray([vec1, vec2]), np.asarray([val1, val2])


def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    return (X - np.mean(X, axis=0)).dot(vecs.T)


def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """
    return np.sum(eigenvalues) / total_variance(X)


def plot_graph(data_xy, title, file_name):
    """
    Plot a scatter graph of the data.
    """
    plt.figure(figsize=(17, 12))
    plt.suptitle(title, fontsize=25)

    for i, language in enumerate(languages):
        plt.scatter(data_xy[0, i], data_xy[1, i], c=language_color[language])
        plt.annotate(language, (data_xy[0, i], data_xy[1, i]))

    plt.show() if PLOT_SAVE else plt.savefig(file_name)
    return


def plot_PCA():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of PCA on languages data.
    """
    data_xy = vec.dot(data.T)  # project data back

    plot_graph(data_xy,
               "PCA (Explained Variance Ratio = %.3f)" % variance_ratio,
               "PCA.jpg")
    return


def plot_MDS():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of MDS (Multidimensional scaling) on languages data.

    Use sklearn.manifold.MDS and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets.
    """
    mds_embedding = MDS(n_components=2,
                        random_state=0,
                        dissimilarity='precomputed')
    data_xy = mds_embedding.fit_transform(data_dist).T

    plot_graph(data_xy, "MDS", "MDS.jpg")
    return


def plot_TSNE():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of TSNE (T-distributed Stochastic Neighbor Embedding) 
    on languages data.

    Use sklearn.manifold.TSNE and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets.
    """
    tsne_embedding = TSNE(n_components=2,
                          perplexity=30,
                          init='random',
                          random_state=0,
                          metric='precomputed',
                          square_distances=True,
                          learning_rate='auto')
    data_xy = tsne_embedding.fit_transform(data_dist).T

    plot_graph(data_xy, "TSNE", "TSNE.jpg")
    return


if __name__ == "__main__":
    global data, data_dist, languages, variance_ratio, language_color

    data_dict = read_data(3)
    data_dist = distance_mat(data_dict)
    data, languages = prepare_data_matrix(data_dict)

    # eigenvectors and corresponding eigenvalues
    vec, val = power_iteration_two_components(data)
    variance_ratio = explained_variance_ratio(data, vec, val)

    language_color = {}
    languages_unique = list(set(languages))
    for i, language in zip(
            random.sample(range(len(color_palette)), len(languages_unique)),
            languages_unique):
        language_color[language] = color_palette[i]

    plot_PCA()
    plot_MDS()
    plot_TSNE()
