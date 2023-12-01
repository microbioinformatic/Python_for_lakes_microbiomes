import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid, KFold
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mode
from utils import entropy
from hdbscan import validity_index as DBCV
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from IPython.core.debugger import set_trace


def metrics_formatting(data, clusters=None, metrics=None, choose_partition_by = 'dbcv'):
    
    '''
    Formats metrics dict, choosing the best partition via `choose_partition_by` criterion
    data = pca_cluster_metrics['AGP_o']['HDBSCAN']
    metrics = ['dbcv', 'ps', 'entropy']
    '''
    
    # cleaning empty results
    for n_clusters, partition_metrics in data.copy().items():
        if len(partition_metrics) == 0:
            data.pop(n_clusters)
    data_chosen = defaultdict(dict)
    
    if clusters is not None:
        for n_clusters, partition_metrics in clusters.copy().items():
            if len(partition_metrics) == 0:
                clusters.pop(n_clusters)
        clusters_chosen = defaultdict(dict)
    
    for n_clusters, partition_metrics in data.items():
        best_partition = partition_metrics[0]
        best_partition_index = 0
        for i,partition in enumerate(partition_metrics):
            if partition[choose_partition_by] > best_partition[choose_partition_by]:
                best_partition = partition
                best_partition_index=i
        if metrics is not None:
            data_chosen[n_clusters] = {k:best_partition[k] for k in metrics}
        else:
            data_chosen[n_clusters] = best_partition
        
        if clusters is not None:
            clusters_chosen[n_clusters] = clusters[n_clusters][best_partition_index]
    return (data_chosen, clusters_chosen) if clusters is not None else data_chosen


def plot_clustering_scatter(metrics_df, 
                            x_metric_name, 
                            y_metric_name, 
                            coloring_metric_name, 
                            title=None,
                            y_threshold=None,
                            x_threshold=None,
                            x_hue_line=None,
                            y_hue_line=None
                            ):
    
    ABBV2NAME = {'dbind':'Davies-Bouldin index',
                'silh':'Silhouette score',
                'dbcv':'DBCV',
                'ch':'Calinski-Harabasz score',
                'ps':'Prediction Strength',
                'entropy': 'Entropy',
                'data_used': 'Data percentage'}
    
    '''
    Creates 2D scatter plot given dataframe
    metrics_df: pd.DataFrame
    x_metric_name: str - abscissa column name
    y_metric_name: str - ordinate column name
    coloring_metric_name: str - columns name used for coloring
    '''
    
    y = metrics_df[y_metric_name]
    x = metrics_df[x_metric_name]
    plt.figure(figsize=(5,5), dpi=300)
    plt.scatter(x,y,
             alpha=0.2,
             c=metrics_df[coloring_metric_name])
    
    cbar = plt.colorbar()
    
    plt.xlabel(ABBV2NAME[x_metric_name])
    plt.ylabel(ABBV2NAME[y_metric_name])
    if title is not None:
        plt.title(title)
    
    cbar.set_label(ABBV2NAME[coloring_metric_name], rotation=270, labelpad=22) 
    
    Y_MIN = y.min()-0.1
    Y_MAX = y.max()+0.1
    X_MIN = x.min()-0.1
    X_MAX = x.max()+0.1
    
    plt.ylim(Y_MIN, Y_MAX)
    plt.xlim(X_MIN, X_MAX)
    
    # horizontal-thresolhd
    if y_threshold is not None:
        
        y_threshold = np.clip(y_threshold, Y_MIN, Y_MAX)
        y_hue_line = np.clip(y_hue_line, Y_MIN, Y_MAX)
        
        plt.hlines(y_threshold, xmin=X_MIN, xmax=X_MAX, linestyle='--', color='blue', 
                   alpha=0.5, label=f'{y_metric_name} threshold')
        plt.fill_between(x=[X_MIN, X_MAX], y1=y_hue_line, y2=y_threshold, color='blue', alpha=0.1)
    
    # vertical-thresolhd
    if x_threshold is not None:
        
        x_threshold = np.clip(x_threshold, X_MIN, X_MAX)
        x_hue_line = np.clip(x_hue_line, X_MIN, X_MAX)
        
        plt.vlines(x_threshold, ymin=Y_MIN, ymax=Y_MAX, linestyle='--', color='orange', alpha=0.5, label=f'{x_metric_name} threshold')
        plt.fill_between(x=sorted([x_hue_line, x_threshold]), y1=Y_MIN, y2=Y_MAX, color='orange', alpha=0.1)
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    plt.show()


def clustering_by_methods(data, methods_dict, precomputed=False, d=None, verbose=False, cluster_perc_threshold=0.01):
    '''
    Perform clustering of `data` for each method and 
    hyperparameters combination inferred from the `methods_dict`
    
    data: np.ndarray [N,d]
    methods_dict: dict of pairs {'class_name':[class, params_dict], ...}
    e.g.
    {
       'HDBSCAN':[HDBSCAN, 
          {'min_cluster_size':[5,10,25,50], 
           'min_samples':[None,5,10,15,20],
           'metric':['precomputed'],
           'core_dist_n_jobs':[1]}
           ],
       'KMedoids':[KMedoids, 
                   {'n_clusters':np.arange(2, 10, 1),
                    'method':['pam'],
                    'metric':['precomputed'],
                    'init':['k-medoids++'],
                    'random_state':[42]}],
       'SpectralClustering':[SpectralClustering_prec, 
                             {'n_clusters':np.arange(2, 10, 1), 
                              'gamma':[0.1, 1, 5, 10, 15, 20, 25, 30],
                              'eigen_tol':[1e-4]}]
    }
    precomputed: bool - whether use precomputed distances, then data treated as pairwise distance matrix [N,N]
    d: dataset dimensionality d (useful when precomputed=True)
    '''
    
    results = {}
    for method_name, [method_class, param_range] in methods_dict.items():
        if verbose:
            print('----------------------------')
            print('Clustering for', method_name)
        
#         try:
        cluster_results = clustering(data, 
                                    method_class, 
                                    param_range,
                                    precomputed,
                                    d=d,
                                    verbose=verbose,
                                    cluster_perc_threshold=cluster_perc_threshold)
#         except:
#             if verbose:
#                 print(f'Error during metrics computation for {method_class.__name__}')
#                 continue

        results[method_name] = cluster_results
    return results


def clustering(dataset, 
               method_class, 
               param_dict, 
               precomputed=False, 
               verbose=False, 
               d=None,
               cluster_perc_threshold=0.01):
    
    '''
    dataset: dataset [N,d] or distance matrix [N,N]
    method_class: class constructor for clustering algorithm
    param_dict: params for class constructor e.g. {'n_clusters':[2,3,4,5], 'metric':['euclidean']}
    precomputed: dataset will be treated as distance matrix [N,N]
    verbose: bool - whether print warning messages
    d: int - The number of features (dimension) of the dataset.
    '''
    
    cluster_results = [] # partition results
    
    for p in ParameterGrid(param_dict):
        
        results = {}
        
        method = method_class(**p)
        pred = method.fit_predict(dataset)
        
        # at least 2 clusters
        # -1:outliers, 0:first cluster, 1:second cluster etc.
        if max(pred) > 0: 
            
            # consider only non-noise clusters
            non_noise_mask = pred != -1
            noise_mask = pred == -1
            
            # if too much noise - continue
            if sum(noise_mask)/len(noise_mask) > 0.4:
                if verbose:
                    print(f'Too much nose, skipping for p={p}, {method_class.__name__}')
                continue
            
            # filter-out small clusters
            abundance_mask = np.zeros(len(pred), dtype=bool)
            # iterating over non-noise classified points
            for k in np.unique(pred[non_noise_mask]): 
                # more than 1% of the data
                if sum(pred==k)/sum(non_noise_mask) > cluster_perc_threshold:
                    abundance_mask[pred==k] = True
                elif verbose:
                    print(f'Small cluster {k} with {sum(pred==k)} items removed')
                    
            mask = abundance_mask
            results['mask'] = abundance_mask
            # all data was separated into small clumps
            data_used = mask.sum()/len(mask)
            significant_clusters = pred[mask]
            
            if data_used < 0.5:
                if verbose:
                    print(f'Too much data were removed!')
                continue
            
            # no outliers left
            assert (significant_clusters >= 0).all()
            # all clusters contain more that 1% of data
            assert Counter(significant_clusters).most_common()[-1][1]/sum(non_noise_mask) > cluster_perc_threshold
            
            unique_clusters = np.unique(significant_clusters)
            n = len(unique_clusters) # number of clusters
            
            # re-numerated unique_clusters labels
            labels = np.zeros((sum(mask)), dtype=int)
            for i,k in enumerate(unique_clusters):
                labels[significant_clusters==k] = i
            results['labels'] = labels
            
            if n > 1:
                
                if precomputed:
                    results['dbind'] = davies_bouldin_score_precomputed(dataset[mask][:,mask], labels)
                    results['silh'] = silhouette_score(dataset[mask][:,mask], labels, metric='precomputed')
                    try:
                        results['dbcv'] = DBCV(dataset[mask][:,mask], labels, metric='precomputed', d=d)
                    except:
                        results['dbcv'] = np.nan
                    
                    results['ps'] = prediction_strength_CV_precomputed(dataset[mask][:,mask], method=None, y=labels)

                else:
                    results['dbind'] = davies_bouldin_score(dataset[mask], labels)
                    results['silh'] = silhouette_score(dataset[mask], labels)
                    try:
                        results['dbcv'] = DBCV(dataset[mask], labels)
                    except:
                        results['dbcv'] = np.nan
                    try:
                        results['ps'] = prediction_strength_CV(dataset[mask], method=None, y=labels) 
                    except:
                        results['ps'] = np.nan
                    
                # data mass distribution
                cl_dist = np.ones(n)
                for i in range(n):
                    cl_dist[i] = sum(labels == i)/sum(mask)

                results['noise_ratio'] = sum(noise_mask)/len(noise_mask)
                results['entropy'] = entropy(cl_dist)
                results['data_used'] = data_used
                results['dist'] = cl_dist

                # for each [n] there may be more than 1 partition!
                cluster_results.append(results)
                
            else:
                if verbose:
                    print(f'No clusters found for p={p}, {method_class.__name__}')
                continue

        # no clusters found
        else:
            if verbose:
                print(f'No clusters found for p={p}, {method_class.__name__}')
            continue
                
    return cluster_results


def davies_bouldin_score_precomputed(D, labels):
    
    '''
    Calculates Davies-Bouldin index for precomputed data D
    D: np.ndarray [N,N] - pairwise distance matrix
    labels: np.ndarray [N,] - clustering labels
    '''
    
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = D.shape
    n_labels = len(le.classes_)
    intra_dists = np.zeros(n_labels)
    centroids = []
    
    for k in range(n_labels):
        mask_k = labels==k
        cluster_k = D[mask_k][:,mask_k] #_safe_indexing(X, labels == k)
        centroid_index = np.argmin(cluster_k.mean(1))
        intra_dists[k] = cluster_k[centroid_index].mean()
        centroid_index = np.arange(len(mask_k))[mask_k][centroid_index]
        centroids.append(centroid_index)
        
    centroid_distances = D[centroids][:,centroids] # kxk matrix

    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists # kxk matrix
    scores = np.max(combined_intra_dists / centroid_distances, axis=1)
    return np.mean(scores)

def prediction_strength(y_pred, y_test):
    '''
    For each pair of test observations 
    that are assigned to the same test cluster, 
    we determine whether they are also assigned 
    to the same cluster based on the train desicion boundary.
    '''
    
    test_clusters = np.unique(y_test)
    counts = []
    
    for k in test_clusters:
        
        # noise cluster
        if k == -1:
            continue
        
        mask = y_test == k
        n_k = mask.sum()
        
        # number of points more than 1 in cluster
        if n_k > 1:

            c = Counter(y_pred[mask])
            count = c.most_common(1)[0][1]
            count = count * (count - 1) # number of pairs that fall in the same cluster given train decision function
            count /= (n_k * (n_k - 1)) # divided by the total number of pairs in cluster

            counts.append(count)
            
    return min(counts) if len(counts) > 0 else 0


def prediction_strength_CV_precomputed(D, method=None, y=None, n_splits=3, knn=5):
    
    '''
    Calculates Prediction Strength for precomputed data D
    D: np.ndarray [N,N] - pairwise distance matrix
    method: clustering method instance e.g. KMeans()
    n_splits: number of folds for cross validation splits
    '''
    
    ps_s = []
    kfold = KFold(n_splits=n_splits, shuffle=True)
    
    for i,(train_index, test_index) in enumerate(kfold.split(D)):
        
        # getting clustering from train data
        D_train = D[train_index][:,train_index]
        
        if method is None:
            y_train = y[train_index] 
        else:
            y_train = method.fit_predict(D_train) 
        
        # getting clustering from test data
        D_test = D[test_index][:,test_index]
        
        if method is None:
            y_test = y[test_index] 
        else:
            y_test = method.fit_predict(D_test)
        
        D_ = D[test_index][:,train_index] 
        y_pred = y_train[np.argsort(D_, axis=1)[:,:knn]]
        y_pred = mode(y_pred, axis=1).mode.flatten()
        
        ps = prediction_strength(y_pred, y_test) # y_train, y_test 
        ps_s.append(ps)
        
    return np.mean(ps_s)


def prediction_strength_CV(X, method=None, y=None, n_splits=3, knn=5):
    
    '''
    Calculates Prediction Strength for  data X
    X: np.ndarray [N,N=d] - dataset
    method: clustering method instance e.g. KMeans()
    n_splits: number of folds for cross validation splits'
    '''
    
    ps_s = []
    kfold = KFold(n_splits=n_splits, shuffle=True)
    
    for i,(train_index, test_index) in enumerate(kfold.split(X)):
        
        # getting clustering from train data
        X_train = X[train_index]
        if method is None:
            y_train = y[train_index] 
        else:
            y_train = method.fit_predict(X_train)
        
        # getting clustering from test data
        X_test = X[test_index]
        if method is None:
            y_test = y[test_index] 
        else:
            y_test = method.fit_predict(X_test)

        clf = KNeighborsClassifier(weights='distance', p=2, n_neighbors=knn) 
        clf.fit(X_train, y_train) # fit decision regions from train data
        y_pred = clf.predict(X_test) # predict test clustering
        
        ps = prediction_strength(y_pred, y_test) # y_train, y_test 
        ps_s.append(ps)
        
    return np.mean(ps_s)