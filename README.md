# Recommender Systems Assignment

CSL7110 - Machine Learning with Big Data

## Overview

This notebook implements various recommendation algorithms on the MovieLens dataset as per the assignment requirements:

- **Part 1:** Content-Based Filtering (TF-IDF, User Profiles)
- **Part 2:** Collaborative Filtering (User-Based, Item-Based)
- **Part 3:** Matrix Factorization (SVD from scratch, Surprise library)
- **Part 4:** Hybrid Model (meta-learning approach)
- **Part 5:** Learning-Based (Neural Network, Reinforcement Learning)
- **Part 6:** Explainability (SHAP, k-NN explanations, LIME)

## Dataset

MovieLens ml-latest-small:
- 100,836 ratings
- 610 users
- 9,742 movies
- Rating scale: 0.5 - 5.0

Dataset files in `data/` folder:
- ratings.csv
- movies.csv
- tags.csv
- links.csv

## Requirements

- Python 3.10 or 3.11
- ~4GB RAM for similarity matrix computations

## Setup

1. Create virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**For non-Mac systems:** Change `tensorflow-macos` and `tensorflow-metal` to just `tensorflow` in requirements.txt

## Running the Notebook

```bash
source .venv/bin/activate
jupyter notebook Recommender_Systems_Assignment.ipynb
```

Run all cells from top to bottom. First run takes longer because it computes similarity matrices. Subsequent runs use cached results.

## Project Structure

```
.
├── Recommender_Systems_Assignment.ipynb   # Main notebook
├── requirements.txt                        # Dependencies
├── README.md                              # This file
├── ANALYSIS.md                            # Results report
├── data/                                  # MovieLens dataset
└── outputs/
    ├── metrics/                           # Evaluation CSVs
    ├── models/                            # Saved models
    ├── plots/                             # Visualizations
    ├── recommendations/                   # Sample outputs
    └── explanations/                      # SHAP/LIME results
```

## Tasks Implemented

| Task | Description | Marks |
|------|-------------|-------|
| 1 | TF-IDF content-based filtering | 20 |
| 2 | User profile based recommendations | 20 |
| 3 | User-based collaborative filtering (Pearson + Cosine) | 20 |
| 4 | Item-based collaborative filtering | 20 |
| 5 | SVD from scratch using numpy/scipy | 20 |
| 6 | SVD with Surprise library + hyperparameter tuning | 20 |
| 7 | Hybrid model (meta-learning) | 10 |
| 8 | Neural network recommender | 40 |
| 9 | Reinforcement learning (ε-Greedy, UCB, Q-Learning) | 40 |
| 10 | SHAP explanations | 10 |
| 11 | k-NN neighborhood explanations | - |
| 12 | LIME explanations | - |
| 13 | Explainability evaluation | - |

## Results Summary

| Model | RMSE | Precision@10 | Recall@10 |
|-------|------|--------------|-----------|
| Hybrid | 0.746 | 0.402 | 0.780 |
| SVD (Surprise) | 0.857 | 0.583 | 0.683 |
| Item-CF | 0.857 | 0.004 | 0.001 |
| NCF Neural Net | 0.897 | 0.559 | 0.676 |
| TF-IDF CBF | 0.923 | 0.539 | 0.656 |

See ANALYSIS.md for detailed results and discussion.

## Dependencies

- numpy, pandas, scipy - data handling
- scikit-learn - ML utilities, metrics
- scikit-surprise - collaborative filtering
- tensorflow - neural network
- shap, lime - explainability
- matplotlib, seaborn - visualization
- tqdm - progress bars

## Notes

- All code is commented and modular
- Random seed = 42 for reproducibility
- Models and intermediate results are cached in outputs/models/
- Evaluation uses RMSE, Precision@K, and Recall@K as specified

## Author

Rahul Agarwal
