import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Sort data by FICO score
df_sorted = df.sort_values('fico_score').reset_index(drop=True)

def quantize_mse(fico_scores, n_buckets):
    from sklearn.cluster import KMeans
    
    # KMeans on FICO scores to find cluster centers
    kmeans = KMeans(n_clusters=n_buckets, random_state=42)
    kmeans.fit(fico_scores.reshape(-1, 1))
    
    centers = np.sort(kmeans.cluster_centers_.flatten())
    
    # Assign boundaries halfway between centers
    boundaries = [(centers[i] + centers[i+1])/2 for i in range(len(centers)-1)]
    boundaries = [fico_scores.min()-1] + boundaries + [fico_scores.max()+1]
    
    # Assign ratings
    ratings = []
    for score in fico_scores:
        for i in range(len(boundaries)-1):
            if boundaries[i] < score <= boundaries[i+1]:
                ratings.append(i+1)  # Lower rating = better score
                break
    
    return boundaries, ratings

def quantize_loglikelihood(fico_scores, defaults, n_buckets):
    scores = np.sort(np.unique(fico_scores))
    n = len(scores)
    
    # Precompute cumulative sums
    cum_n = np.zeros(n+1)
    cum_k = np.zeros(n+1)
    for i, s in enumerate(scores):
        mask = fico_scores == s
        cum_n[i+1] = cum_n[i] + mask.sum()
        cum_k[i+1] = cum_k[i] + defaults[mask].sum()
    
    # Initialize DP tables
    dp = np.full((n+1, n_buckets+1), -np.inf)
    boundaries = np.zeros((n+1, n_buckets+1), dtype=int)
    dp[0,0] = 0
    
    def ll(start, end):
        ni = cum_n[end] - cum_n[start]
        ki = cum_k[end] - cum_k[start]
        if ni == 0 or ki == 0 or ki == ni:
            return 0
        pi = ki / ni
        return ki*np.log(pi) + (ni-ki)*np.log(1-pi)
    
    for j in range(1, n_buckets+1):
        for i in range(1, n+1):
            for k in range(j-1, i):
                val = dp[k,j-1] + ll(k, i)
                if val > dp[i,j]:
                    dp[i,j] = val
                    boundaries[i,j] = k
    
    # Recover bucket boundaries
    b = []
    i = n
    j = n_buckets
    while j > 0:
        b.append(scores[boundaries[i,j]])
        i = boundaries[i,j]
        j -= 1
    b = sorted(b)
    b = [fico_scores.min()-1] + b + [fico_scores.max()+1]
    
    # Assign ratings
    ratings = []
    for score in fico_scores:
        for i in range(len(b)-1):
            if b[i] < score <= b[i+1]:
                ratings.append(i+1)
                break
    
    return b, ratings

# Convert FICO to numpy arrays
fico_scores = df_sorted['fico_score'].values
defaults = df_sorted['default'].values

# Number of desired buckets
n_buckets = 5

# MSE approach
mse_boundaries, mse_ratings = quantize_mse(fico_scores, n_buckets)
df_sorted['mse_rating'] = mse_ratings

# Log-likelihood approach
ll_boundaries, ll_ratings = quantize_loglikelihood(fico_scores, defaults, n_buckets)
df_sorted['ll_rating'] = ll_ratings

print("MSE boundaries:", mse_boundaries)
print("Log-likelihood boundaries:", ll_boundaries)
print(df_sorted[['fico_score','default','mse_rating','ll_rating']].head(20))
