import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy import stats

def SVM_multiclass(feat_train, labels_train, feat_test, C=1.0, kernel = 'linear', max_iteration = 1000, degree=3):
    """Support Vector Machine for a multiclass problem(one vs rest).

    :param feat_train: array[n_samples,n_feats]
        Array containing the training samples
    :param labels_train: array[n_samples]
        Array containing the label for each training sample
    :param feat_test: array[n_samples,n_feats]
        Array containing the test samples
    :param kernel: string(default 'linear')
        The type of kernel
    :param max_iteration: int(default 1000)
        Max iteration
    :param degree: int(default 3)
        The polynomial order of the kernel. Only for the 'poly' kernel
    :return: array[n_samples]
        The classes predicted from feat_test
    """
    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test = scaler.transform(feat_test)
    n_classes = np.unique(labels_train)

    models = list()
    for i in range(n_classes.shape[0]):
        models.append(SVC(C=C, kernel=kernel, max_iter=max_iteration, degree=degree, probability=True, class_weight="balanced", random_state=42))
        models[-1].fit(feat_train, labels_train==i)

    predicted_scores = []
    for i in range(n_classes.shape[0]):
        predicted_scores.append(models[i].predict_proba(feat_test)[:, 1])
    predicted_scores = np.asarray(predicted_scores)
    predicted = np.argmax(predicted_scores, axis=0)

    return predicted

# METRIC=‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘wminkowski’
def KNN_classifier(feat_train, labels_train, feat_test, K=3, metric='euclidean'):
    """ K-Nearest Neighborhood

    :param feat_train: array[n_samples,n_feats]
        Array containing the training samples
    :param labels_train: array[n_samples]
        Array containing the label for each training sample
    :param feat_test: array[n_samples,n_feats]
        Array containing the test samples
    :param K: int(default 3)
        The number of neighboor
    :param metric: string(default 'euclidean')
        The metric to compute the distances
    :return: array[n_samples]
        The classes predicted from feat_test
    """

    # Compute the distances
    D = cdist(feat_train,feat_test,metric=metric)
    D = np.argsort(D, axis=0)
    k_neigh = D[:K,:]

    l_neigh = labels_train[k_neigh]
    predicted = stats.mode(l_neigh,axis=0)[0][0]

    return predicted



