
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

def cosine_sim(A, B):
    v = cosine_similarity(A, B).mean()
    return float(v)

def pairwise_distance(A, B):
    from numpy.linalg import norm
    m = min(len(A), len(B))
    A = A[:m]; B=B[:m]
    return float(np.mean(norm(A-B, axis=1)))

def log_cluster(A, B, k=5, seed=42):
    kmA = KMeans(n_clusters=k, random_state=seed).fit(A)
    kmB = KMeans(n_clusters=k, random_state=seed).fit(B)
    # compare label histograms
    hA,_ = np.histogram(kmA.labels_, bins=np.arange(k+1))
    hB,_ = np.histogram(kmB.labels_, bins=np.arange(k+1))
    p = (hA+1)/float(hA.sum()+k)
    q = (hB+1)/float(hB.sum()+k)
    return float(np.log( (p*q).sum() ))

def kld_marginal(A, B, bins=20):
    # flatten
    a = A.flatten(); b=B.flatten()
    ha,_ = np.histogram(a, bins=bins, density=True)
    hb,_ = np.histogram(b, bins=bins, density=True)
    ha += 1e-8; hb += 1e-8
    ha /= ha.sum(); hb /= hb.sum()
    return float(entropy(ha, hb))
