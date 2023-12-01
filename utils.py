import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform

import torch
import numpy as np 

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from scipy.stats import multivariate_normal
RES = 100


def classify_enterotypes(X):
    c_s = ['g__Bacteroides', 'g__Prevotella', 'g__Ruminococcus']

    y = -np.ones((X.shape[0]))
    
    ent_1_mask_bact = (X['g__Bacteroides'] > 0.2 ) * (X['g__Prevotella'] < 0.05 ) * (X['g__Ruminococcus'] < 0.015 )
    ent_2_mask_bact = (X['g__Bacteroides'] <= 0.2 ) * (X['g__Prevotella'] > 0.05 ) * (X['g__Ruminococcus'] < 0.020 )    
    ent_3_mask_bact = (X['g__Bacteroides'] < 0.2 ) * (X['g__Prevotella'] < 0.05 ) * (X['g__Ruminococcus'] < 0.07 )    
    
    y[ent_1_mask_bact] = 0
    y[ent_2_mask_bact] = 1    
    y[ent_3_mask_bact] = 2    
    
    return y

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df


def mvn_pdf(X, mean, cov):
    '''
    X - [:, d]
    mean - [1,d]
    cov - [d,d]
    '''
    det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    d = cov.shape[0]
    return (1./(np.sqrt( det*(2*np.pi)**d )))*np.exp(-0.5*(X-mean)@cov_inv@(X-mean).T)

def get_density(X,y):
    
    # calculate clusters means and covariance matrices
    means = []
    cov = []
    mix_comp = []
    for y_k in np.unique(y):
        y_k_mask = y == y_k
        X_k = X[y_k_mask]
        X_k_cent = X_k.copy()
        X_k_cent -= X_k.mean(0, keepdims=True) 
        C = X_k_cent.T@X_k_cent / (X_k_cent.shape[0]-1)

        means.append(X_k.mean(0))
        cov.append(C)
        mix_comp.append(y_k_mask.sum() / len(y))

    means = np.array(means)
    cov = np.array(cov)
    mix_comp = np.array(mix_comp)

    x_grid = np.linspace(X[:,0].min(), X[:,0].max(), num=RES)
    y_grid = np.linspace(X[:,1].min(), X[:,1].max(), num=RES)
    XX = np.stack(np.meshgrid(x_grid, y_grid),axis=-1).reshape(-1,2)

    clusters_likelihoods = []
    for pi_k, mean_k, cov_k in zip(mix_comp, means, cov):

        XX_likelihood = []
        for XX_i in tqdm(XX): 

            XX_i_likelihood = multivariate_normal.pdf(XX_i, mean=mean_k, cov=cov_k)
            XX_likelihood.append(XX_i_likelihood)

        XX_likelihood = np.array(XX_likelihood)
        XX_likelihood = XX_likelihood.reshape(100,100)#[::-1,:]
#         XX_likelihood /= XX_likelihood.sum()

        clusters_likelihoods.append(XX_likelihood)

    clusters_likelihoods = np.array(clusters_likelihoods)
    
    return clusters_likelihoods

def coord_to_pix(X, n_pix=100):
    
    a = X.min()
    b = X.max()
    
    X_ = (X - a)/(b-a+1e-3)
    return X_*n_pix



def l1_normalized_error_torch(y, y_pred):
    '''
    Absolute Percentage Error
    y: torch tensor [N,d]
    y_pred: torch tensor [N,d]
    '''
    return torch.norm(y_pred - y, dim=1, p=1) / (torch.norm(y, dim=1, p=1) + 1e-7)

def l1_normalized_error(y, y_pred):
    '''
    Absolute Percentage Error
    y: np.ndarray [N,d]
    y_pred: np.ndarray [N,d]
    '''
    return np.linalg.norm(y_pred - y, axis=1, ord=1) / (np.linalg.norm(y, axis=1, ord=1) + 1e-7)


def KNN_MAE(X, Z, averaging=None, weights='uniform', n_neighbors=4):
    '''
    Calculate K-Nearest Neighbours Leave-One-Out Median Absolute Percentage Error
    X: np.ndarray [N,d1] - data
    Z: np.ndarray [N,d2] - embedding (d2 <= d)
    averaging: Mean or Median Absolute Percentage Error
    weights: weights for KNN prediction calculation
    '''
    nn = NearestNeighbors(n_neighbors=n_neighbors+1)
    nn.fit(Z)
    Z_kdistance, Z_kneighbors = nn.kneighbors(Z)
            
    if weights=='uniform':
        X_pred = X[Z_kneighbors[:,1:]].mean(1)
    elif weights=='distance':
        D = Z_kdistance[:,1:] + 1e-9
        D = 1./D # create weights of linear combination
        D = D/D.sum(1)[:,None]
        D = D[:,:,None]
        X_pred = X[Z_kneighbors[:,1:]] * D
        X_pred = X_pred.sum(1)
    else:
        raise KeyError('Wrong weights type!')
            
    error = l1_normalized_error(X, X_pred)
     
    if averaging=='mean':
        mae = np.mean(error)
    elif averaging=='median':
        mae = np.percentile(error,50)
    elif averaging==None:
        mae = error 
    else:
        raise KeyError('Wrong averaging attribute!')
        
    return mae


def coranking_matrix_precomputed(D, Z):
    '''
    Generate a co-ranking matrix from two data frames of high and low
    dimensional data like `coranking_matrix`, but for pre-computed distance matrices
    D: pairwise distance matrices for original data 
    Z: pairwise distance matrices for embedding data 
    '''
 
    n = D.shape[0]
    high_distance = D
    low_distance = squareform(pdist(Z))

    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
    low_ranking = low_distance.argsort(axis=1).argsort(axis=1)

    Q, xedges, yedges = np.histogram2d(high_ranking.flatten(),
                                       low_ranking.flatten(),
                                       bins=n)

    Q = Q[1:, 1:]  # remove rankings which correspond to themselves
    return Q

def coranking_matrix(high_data, low_data):
    # from https://github.com/samueljackson92/coranking/blob/master/coranking/_coranking.py
    """Generate a co-ranking matrix from two data frames of high and low
    dimensional data.
    :param high_data: DataFrame containing the higher dimensional data.
    :param low_data: DataFrame containing the lower dimensional data.
    :returns: the co-ranking matrix of the two data sets.
    """
    n, m = high_data.shape
    high_distance = squareform(pdist(high_data))
    low_distance = squareform(pdist(low_data))

    high_ranking = high_distance.argsort(axis=1).argsort(axis=1)
    low_ranking = low_distance.argsort(axis=1).argsort(axis=1)

    Q, xedges, yedges = np.histogram2d(high_ranking.flatten(),
                                       low_ranking.flatten(),
                                       bins=n)

    Q = Q[1:, 1:]  # remove rankings which correspond to themselves
    return Q

    
def entropy(x):
    '''
    Calculates Shannon entropy for a 1-dimensional pdf
    x: [T,] - distribution pdf (np.sum(x) should be 1)
    '''
    return -(x*np.log(x)).sum()

    
def project_pca(data, ev_threshold=0.99, whiten=True, centering=True, random_state=42):
    
    '''
    Compute PCA projection that preserves `ev_threshold` explained variance
    data: np.ndarray [N,d] - dataset
    ev_threshold: explained variance threshold
    '''
    
    
    if centering:
        data_centered = data - data.mean(0)[None,...]
    else:
        data_centered = data
    pca = PCA(svd_solver='full', whiten=whiten, random_state=random_state)
    pca.fit(data_centered)
    explained_variance = pca.explained_variance_ratio_
    d = len(explained_variance)
    
    ev_num = np.arange(1,d+1)[np.cumsum(explained_variance) >= ev_threshold]
    ev_num = ev_num[0]
    pca_proj = PCA(n_components=ev_num, whiten=whiten, svd_solver='full',random_state=random_state)
    data_projected = pca_proj.fit_transform(data_centered)
    mae = np.percentile(l1_normalized_error(pca_proj.inverse_transform(data_projected), data_centered), 50)
    
    return data_projected, pca, pca_proj, mae


def NPR(X, Z, k=21):
    '''
    Neighbourhood preservation ratio
    X: np.ndarray [N,d1] - data
    Z: np.ndarray [N,d2] - embedding (d2 <= d)
    '''
    _, neigborhood_X = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(X)
    _, neigborhood_Z = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z)
    n = X.shape[0]
    npr = 0
    for i in range(n):
        npr += np.intersect1d(neigborhood_X[i], neigborhood_Z[i]).shape[0]
    npr_normed = npr / (k * n)
    return npr_normed


def transform(method, X, dim, parameters, scorer):
    '''
    Given manifold learning method with given parameters
    computes embedding of dataset. Evaluates the embedding quality with 
    scorer function.
    
    method: class - manifold learning method
    X: np.ndarray - dataset
    dim: dimensionality to reduce the original data to
    parameters: dict - manifold learning hyperparameters
    scorer: function that given original data X and embedding Z returns embedding quality
    '''

    model_inst = method(n_components=dim, **parameters)
    Z = model_inst.fit_transform(X)
    score = scorer(X,Z) 
    return score


def calculate_Q_metrics(X, Z, D=None, precomputed=False):
    '''
    Calculates co-ranking matrix based metrics Q_loc and Q_glob
    X: np.ndarray [N,d1] - data
    Z: np.ndarray [N,d2] - embedding (d2 <= d)
    D: pairwise distance matrices for original data if precomputed=True
    '''
    
    if precomputed:
        Q = coranking_matrix_precomputed(D, Z)
    else:
        Q = coranking_matrix(X, Z)
    
    N = X.shape[0]
    UL_cumulative = 0 
    Q_k = []
    LCMC_k = []
    
    for k in range(Q.shape[0]):
        r = Q[k:k+1,:k+1].sum()
        c = Q[:k,k:k+1].sum()

        UL_cumulative += (r+c)
        Qnk = UL_cumulative/((k+1)*N) 

        Q_k.append(Qnk)
        LCMC_k.append(Qnk - ((k+1)/(N-1)))

    k_max = np.argmax(LCMC_k) + 1

    Q_loc = (1./(k_max))*np.sum(Q_k[:k_max])
    Q_glob = (1./(N-k_max-1))*np.sum(Q_k[k_max:]) 
    
    return [Q_loc, Q_glob]