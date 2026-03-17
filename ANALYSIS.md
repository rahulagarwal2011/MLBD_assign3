# Recommender Systems Assignment Report

## CSL7110 - Machine Learning with Big Data

**Student:** Rahul Agarwal

---

## Dataset

Using MovieLens ml-latest-small dataset:
- 100,836 ratings from 610 users on 9,742 movies
- Rating scale: 0.5 to 5.0 stars
- Sparsity: ~98.3% (most user-movie pairs have no rating)

---

## Part 1: Content-Based Filtering (20 marks)

### Task 1: TF-IDF Based Recommendation

**Implementation:**
- Extracted genres from movies.csv as text descriptions
- Computed TF-IDF vectors using sklearn's TfidfVectorizer
- Calculated cosine similarity between movie vectors
- Built recommendation function that returns top-N similar movies

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 0.923 |
| Precision@10 | 0.539 |
| Recall@10 | 0.656 |

**Sample Output** (for "Toy Story (1995)"):
1. Toy Story 2 (1999) - 1.000
2. Antz (1998) - 0.943
3. Emperor's New Groove (2000) - 0.912
4. Monsters, Inc. (2001) - 0.898
5. Finding Nemo (2003) - 0.887

The recommendations make sense - all are animated family movies similar to Toy Story.

### Task 2: User-Profile Based Recommendations

**Implementation:**
- Built user profiles as weighted average of TF-IDF vectors
- Weights = user's ratings for each movie
- Computed similarity between user profile and all movie vectors
- Ranked movies by similarity score

**Formula used:**
```
P_u = Σ(r_u,m * f_m) / Σ(r_u,m)
```
where r_u,m is user u's rating for movie m, and f_m is the TF-IDF vector.

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 1.303 |
| Precision@10 | 0.502 |
| Recall@10 | 0.640 |

Higher RMSE than TF-IDF alone but this approach personalizes recommendations based on user history.

---

## Part 2: Collaborative Filtering (20 marks)

### Task 3: User-Based Collaborative Filtering

**Implementation:**
- Built user-item rating matrix
- Computed user-user similarity using both Pearson correlation and Cosine similarity
- Predicted ratings using weighted average of K nearest neighbors
- Tested with K = 5, 10, 20, 50

**Results:**

| Similarity | K | RMSE | Precision@10 | Recall@10 |
|------------|---|------|--------------|-----------|
| Pearson | 5 | 0.982 | 0.056 | 0.087 |
| Pearson | 10 | 0.958 | 0.026 | 0.036 |
| Pearson | 20 | 0.953 | 0.012 | 0.014 |
| Pearson | 50 | 0.953 | 0.003 | 0.004 |
| Cosine | 5 | 0.981 | 0.052 | 0.083 |
| Cosine | 10 | 0.957 | 0.024 | 0.035 |
| Cosine | 20 | 0.953 | 0.011 | 0.015 |
| Cosine | 50 | 0.953 | 0.003 | 0.005 |

**Observations:**
- RMSE improves with more neighbors (K=50 best for RMSE)
- But Precision/Recall decrease with more neighbors (K=5 best for ranking)
- Theres a tradeoff between prediction accuracy and ranking quality
- Pearson and Cosine give similar results, Pearson slightly better for handling user rating biases

### Task 4: Item-Based Collaborative Filtering

**Implementation:**
- Computed item-item similarity matrix using cosine similarity
- For each user, predicted ratings based on similar items they've rated
- Used weighted average where weights = similarity scores

**Results (K=20):**
| Metric | Value |
|--------|-------|
| RMSE | 0.857 |
| Precision@10 | 0.004 |
| Recall@10 | 0.001 |

**Comparison with User-Based CF:**
- Item-CF has better RMSE (0.857 vs 0.953)
- But lower ranking metrics

**Answer to conceptual question:**
Yes, Item-based CF is generally faster and more memory efficient in production because:
1. Item similarities are more stable over time (items dont change, but user preferences do)
2. Number of items is usually smaller than number of users in real systems
3. Item similarity matrix can be precomputed and cached
4. User-based CF needs to recompute similarities as new users join

---

## Part 3: Matrix Factorization (20 marks)

### Task 5: SVD from Scratch

**Implementation:**
- Used scipy.sparse.linalg.svds for truncated SVD
- k = 50 latent factors
- Reconstructed rating matrix as R ≈ U * Σ * V^T
- Predicted missing ratings from reconstructed matrix

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 0.918 |
| Precision@10 | 0.047 |
| Recall@10 | 0.059 |

### Task 6: SVD with Surprise Library

**Implementation:**
- Used GridSearchCV to tune hyperparameters
- Tested n_factors: [50, 100, 150], n_epochs: [20, 30, 50], lr_all: [0.002, 0.005, 0.01], reg_all: [0.02, 0.05, 0.1]

**Best Parameters Found:**
- n_factors: 100
- n_epochs: 20
- lr_all: 0.005
- reg_all: 0.02

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 0.857 |
| Precision@10 | 0.583 |
| Recall@10 | 0.683 |

**Comparison with Task 5:**
Surprise SVD is much better because:
- Hyperparameter tuning finds optimal settings
- Built-in regularization prevents overfitting
- More efficient implementation

---

## Part 4: Hybrid Model (10 marks)

### Task 7: Hybrid Recommendation

**Implementation:**
Used meta-learning approach with Ridge regression to combine:
- CBF score (TF-IDF similarity)
- CF score (item-based prediction)
- Movie popularity (average rating)
- User bias (user's average rating)

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 0.746 |
| Precision@10 | 0.402 |
| Recall@10 | 0.780 |

**Comparison:**
| Model | RMSE |
|-------|------|
| Hybrid | 0.746 |
| CF only | ~0.86 |
| CBF only | ~1.30 |

The hybrid model achieves the best RMSE by combining signals from both approaches.

**Cold-Start Analysis:**
- For users with few ratings, CBF features contribute more
- For active users, CF features dominate
- The meta-model learns to weight features appropriately

---

## Part 5: Learning-Based Recommenders (40 marks)

### Task 8: Neural Network Content-Based Filtering

**Architecture:**
```
User Input → Dense(64) → Dropout(0.3) → Dense(32) → User Embedding
Movie Input → Dense(64) → Dropout(0.3) → Dense(32) → Movie Embedding
Concatenate → Dense(64) → Dropout(0.3) → Dense(1) → Predicted Rating
```

**Training:**
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Batch size: 512
- Epochs: 50 (with early stopping, patience=5)

**Results:**
| Metric | Value |
|--------|-------|
| RMSE | 0.897 |
| Precision@10 | 0.559 |
| Recall@10 | 0.676 |

**Comparison with TF-IDF:**
The neural network captures more complex patterns than simple TF-IDF similarity, but the improvement is modest on this dataset. Neural nets typically need more data to really outperform traditional methods.

### Task 9: Reinforcement Learning

**Implemented three algorithms:**

1. **ε-Greedy (ε=0.1)**
   - 90% exploit (recommend highest estimated reward)
   - 10% explore (random recommendation)

2. **UCB (Upper Confidence Bound)**
   - Balances exploration and exploitation
   - Prioritizes less-explored items with potential high reward

3. **Q-Learning**
   - State: user context
   - Action: movie recommendation
   - Reward: rating >= 4 is positive, < 4 is negative
   - Update rule: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]

**Results:**
| Algorithm | Total Reward | Explorations |
|-----------|--------------|--------------|
| ε-Greedy | 62,631 | 4,941 (9.9%) |
| UCB | 3,976 | 9,724 (19.4%) |
| Q-Learning | 155,758 | 9,982 (10.0%) |

**Observations:**
- Q-Learning achieves highest total reward over 100k episodes
- UCB explores more but converges slower
- ε-Greedy is simple but effective

**Comparison with traditional models:**
RL optimizes for long-term engagement rather than immediate accuracy. Traditional models (SVD, CF) optimize for rating prediction, while RL can learn to balance exploration of new content with exploitation of known preferences.

---

## Part 6: Explainability (10 marks)

### Task 10: Feature-Based Explanations (SHAP)

Used SHAP to explain the hybrid model predictions.

**Top features by importance:**
1. User bias (user's typical rating behavior)
2. CF score (collaborative filtering prediction)
3. Movie popularity
4. CBF score (content similarity)

### Task 11: Neighborhood-Based Explanations

For collaborative filtering, explanations are based on similar users/items.

Example: "This movie was recommended because users similar to you also rated it highly."

The k-NN approach provides intuitive explanations that users can understand.

### Task 12: LIME Explanations

Used LIME for local interpretable explanations of the hybrid model.

**Sample explanation:**
- `user_bias > 3.80`: -1.39 (user rates high, so prediction adjusted down)
- `cf_score 3.58-3.86`: +0.66 (CF predicts moderate rating)
- `popularity > 3.92`: +0.57 (popular movie, slight boost)

### Task 13: Evaluating Explainability

**Do explanations make recommendations clearer?**
Yes - SHAP and LIME show which features drive predictions, helping users understand why a movie was recommended.

**Do explanations reveal biases?**
Yes - we can see that:
- Popular movies get boosted regardless of user preference
- User bias strongly influences predictions (users who rate high get higher predictions)
- CF scores dominate for users with enough history

---

## Summary of Results

| Model | RMSE | Precision@10 | Recall@10 |
|-------|------|--------------|-----------|
| **Hybrid** | **0.746** | 0.402 | **0.780** |
| Item-CF | 0.857 | 0.004 | 0.001 |
| SVD (Surprise) | 0.857 | **0.583** | 0.683 |
| NCF Neural Net | 0.897 | 0.559 | 0.676 |
| SVD (numpy) | 0.918 | 0.047 | 0.059 |
| TF-IDF CBF | 0.923 | 0.539 | 0.656 |
| User-CF | 0.953 | 0.012 | 0.014 |
| User-Profile CBF | 1.303 | 0.502 | 0.640 |

**Key Findings:**
1. Hybrid model achieves best RMSE by combining multiple signals
2. SVD (Surprise) has best precision due to optimized hyperparameters
3. Content-based methods are useful for cold-start scenarios
4. Neural networks need more data to significantly outperform matrix factorization
5. RL provides a different optimization objective (engagement vs accuracy)

---

## Files Submitted

- `Recommender_Systems_Assignment.ipynb` - Main notebook with all tasks
- `README.md` - Setup and running instructions
- `ANALYSIS.md` - This report
- `requirements.txt` - Python dependencies
- `data/` - MovieLens dataset
- `outputs/` - Generated metrics, models, plots, and explanations

---

