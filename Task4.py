import pandas as pd
import numpy as np

df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

# Aggregate by FICO score
grouped = df.groupby("fico_score")["default"].agg(["count","sum"]).reset_index()
fico = grouped["fico_score"].values
n_i = grouped["count"].values
k_i = grouped["sum"].values

m = len(fico)   # ~550 instead of 10,000

# Prefix sums
N = np.zeros(m+1)
K = np.zeros(m+1)
for i in range(m):
    N[i+1] = N[i] + n_i[i]
    K[i+1] = K[i] + k_i[i]

# Log-likelihood of bucket [i, j)
def bucket_ll(i, j):
    n = N[j] - N[i]
    k = K[j] - K[i]
    if k == 0 or k == n:
        return 0
    p = k / n
    return k*np.log(p) + (n-k)*np.log(1-p)

# Precompute bucket likelihoods
LL = np.zeros((m, m))
for i in range(m):
    for j in range(i+1, m+1):
        LL[i,j-1] = bucket_ll(i,j)

# Number of ratings
R = 10

dp = np.full((R+1, m), -np.inf)
cut = np.zeros((R+1, m), dtype=int)

# Base case
for j in range(m):
    dp[1,j] = LL[0,j]

# DP
for r in range(2, R+1):
    for j in range(r-1, m):
        best = -np.inf
        best_i = 0
        for i in range(r-1, j+1):
            val = dp[r-1,i-1] + LL[i,j]
            if val > best:
                best = val
                best_i = i
        dp[r,j] = best
        cut[r,j] = best_i

# Recover boundaries
cuts = []
j = m-1
for r in range(R, 1, -1):
    i = cut[r,j]
    cuts.append(fico[i])
    j = i-1

cuts = sorted(cuts)

print("Optimal FICO boundaries:")
print(cuts)

# Rating map
def fico_to_rating(f):
    for i,b in enumerate(cuts):
        if f < b:
            return R - i
    return 1
