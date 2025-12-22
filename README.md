# Intelligence-Informations-Retrieval-HWs
IIR home woreks in semester 1 of master course in University of Tehran

# Intelligent Information Retrieval – Assignment 4 (IIR‑CA4)

**Course:** Intelligent Information Retrieval  
**Assignment:** CA4 – Word Association, Word Embeddings, and Learning to Rank  
**Dataset:** Cranfield Aeronautics Collection  
**Implementation:** Python  
**Date:** December 22, 2025  
**Calendar:** 2025‑12‑22 (Gregorian) | 1404/10/01 (Jalali)

---

## 📌 Overview

This repository contains the complete implementation and analysis for **Intelligent Information Retrieval – Assignment 4 (IIR‑CA4)**.  
The assignment explores three core pillars of modern IR systems:

1. **Word Association**  
   - Syntagmatic relations using **Mutual Information (MI)**  
   - Paradigmatic relations using **Pseudo‑Documents**

2. **Word Embeddings**  
   - Semantic representation using **GloVe**
   - Dimensionality reduction and visualization with **PCA**

3. **Learning to Rank (LTR)**  
   - Pointwise, Pairwise, and Listwise paradigms  
   - Metric‑driven optimization using **LambdaRank**

All experiments are conducted on the **Cranfield Dataset**, a classical benchmark collection in Information Retrieval, widely used for evaluating retrieval models in scientific and technical domains.

---

## 📚 Dataset: Cranfield Collection

The **Cranfield Collection** is a standard IR test collection consisting of:

- ~1,400 aeronautical research abstracts  
- 225 user queries  
- Expert relevance judgments (graded relevance)

**Domain:** Aerodynamics, fluid mechanics, aircraft design  

**Why Cranfield matters:**  
The dataset is small yet highly technical, making it ideal for studying:
- Vocabulary mismatch
- Term co‑occurrence patterns
- Ranking effectiveness under limited data

**Reference:**  
https://ir-datasets.com/cranfield.html  

---

## 🔗 1. Word Association

Word association is used to uncover latent semantic structures in the corpus beyond simple term frequency. Two complementary relationships are considered: **Syntagmatic** and **Paradigmatic**.

---

### 1.1 Syntagmatic Relations (Mutual Information)

**Definition:**  
Syntagmatic relations capture *co‑occurrence* patterns — words that frequently appear together in context (the “AND” relationship).

**Example:**  
“boundary” AND “layer” → *boundary layer*

#### Method: Mutual Information (MI)

Mutual Information measures how much more often two words occur together compared to chance:

I(x, y) = log ( P(x, y) / (P(x) · P(y)) )

**Implementation Logic:**
- Build a word co‑occurrence matrix using a sliding context window
- Compute joint and marginal probabilities
- Apply smoothing to avoid zero probabilities
- Rank word pairs by MI score

**Cranfield-Specific Findings:**
- High‑MI pairs identify technical collocations:
  - *Mach – number*
  - *Boundary – layer*
  - *Pressure – distribution*

**IR Significance:**
- Improves phrase detection
- Enhances term weighting in retrieval models

**Key Reference:**  
Syntagmatic and Paradigmatic Associations in IR (Springer, 2011)  
https://link.springer.com/chapter/10.1007/978-3-642-18991-3_54  

---

### 1.2 Paradigmatic Relations (Pseudo‑Documents)

**Definition:**  
Paradigmatic relations capture *substitutability* — words that appear in similar contexts but may not co‑occur directly (the “OR” relationship).

**Example:**  
“airfoil” OR “wing”

#### Method: Pseudo‑Documents

A **pseudo‑document** for a word is constructed by aggregating all its surrounding context words across the corpus.

**Implementation Steps:**
1. For each target word, collect neighboring words within a context window
2. Treat this aggregated context as a high‑dimensional vector
3. Compute **Cosine Similarity** between vectors

**Cranfield-Specific Examples:**
- *airfoil* ↔ *wing*
- *velocity* ↔ *speed*

**IR Significance:**
- Core mechanism for **Query Expansion**
- Improves recall by resolving vocabulary mismatch

**Key Reference:**  
Paradigmatic Relation Discovery (UIUC)  
https://aclanthology.org/C02-1007.pdf  

---

## 🧠 2. Word Embeddings

### 2.1 GloVe (Global Vectors for Word Representation)

**GloVe** is an unsupervised embedding model that learns dense vector representations from **global co‑occurrence statistics**.

**Core Insight:**  
Semantic meaning is encoded in the *ratio* of co‑occurrence probabilities rather than raw counts.

**Formal Model (Pennington et al., 2014):**  
The dot product of word vectors approximates the logarithm of word co‑occurrence counts.

**Why GloVe for IR?**
- Addresses vocabulary mismatch
- Captures global corpus semantics
- Particularly effective for technical domains like aeronautics

**Reference:**  
Stanford NLP – GloVe Project (2014)  
https://nlp.stanford.edu/projects/glove/  

---

### 2.2 Embedding Lookup and Semantic Queries

**Implementation Highlights:**
- Pre‑trained GloVe vectors loaded using `gensim`
- Cosine similarity for nearest‑neighbor search

**Cranfield Semantic Neighbors:**
- *supersonic* → hypersonic, transonic, mach
- *airfoil* → wing, camber, chord
- *turbulence* → vortex, instability, flow

**IR Benefit:**  
Enables semantic search beyond exact keyword matching.

---

### 2.3 PCA Visualization

**Purpose:**  
Reduce 100‑dimensional vectors into 2D for qualitative evaluation.

**Tool:**  
`sklearn.decomposition.PCA`

#### Common Cranfield Clusters
- Flight regimes (supersonic, transonic, mach)
- Aero‑structures (airfoil, wing, chord)
- Fluid dynamics (turbulence, viscosity, boundary layer)
- Aerodynamic forces (lift, drag, thrust)

**📊 Placeholder: PCA Scatter Plot**
```
![PCA Visualization of Cranfield Embeddings](images/pca_embeddings.png)
```

**Caption:**  
*“2D PCA projection of GloVe word embeddings reveals distinct semantic clusters within the aerodynamics domain, validating the embedding quality for retrieval tasks.”*

---

## 📈 3. Learning to Rank (LTR)

Learning to Rank formulates retrieval as a supervised machine learning problem using relevance judgments.

---

### 3.1 Pointwise Approach

- Treats ranking as regression/classification
- Predicts absolute relevance scores for each query‑document pair

**Limitation:**  
Ignores relative ordering between documents.

---

### 3.2 Pairwise Approach

- Learns preferences between document pairs
- Minimizes ranking inversions

**Example:**  
If Doc A is more relevant than Doc B, enforce A > B

**Limitation:**  
Does not account for position importance in ranked results.

---

### 3.3 Listwise Approach

- Optimizes the ranking list as a whole
- Directly aligns with IR metrics (MAP, NDCG)

---

### 3.4 LambdaRank

**Key Innovation:**  
Optimizes non‑differentiable ranking metrics by scaling gradients with **ΔNDCG**.

**Why LambdaRank Matters:**
- Penalizes mistakes at the top of rankings
- Directly optimizes retrieval effectiveness

**Typical Implementation:**
- `XGBoost Ranker` or `LightGBM`
- Objective: `rank:ndcg`

**Reference:**  
Burges, C. (2010). *From RankNet to LambdaRank*  
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf  

---

## 📊 Evaluation Metrics

### Mean Average Precision (MAP)
- Measures ranking quality across all recall levels
- Sensitive to the order of relevant documents

### Normalized Discounted Cumulative Gain (NDCG)
- Accounts for graded relevance
- Emphasizes top‑rank accuracy

**Why These Metrics:**  
They are standard for evaluating ranking systems on Cranfield‑style datasets.

---

## 🛠️ Implementation Overview

- **Language:** Python  
- **Key Libraries:**
  - `numpy`, `scikit‑learn`
  - `gensim`
  - `matplotlib`
  - `xgboost` / `lightgbm`

**Notebook:**  
`IIR‑CA4‑Code.ipynb`

The notebook contains:
- Preprocessing pipelines
- MI and pseudo‑document construction
- GloVe embedding lookup
- PCA visualization
- Learning to Rank experiments

---

## ✅ Key Takeaways

- Mutual Information effectively identifies technical collocations in scientific text.
- Pseudo‑documents enable unsupervised synonym discovery for query expansion.
- GloVe embeddings successfully capture domain‑specific semantics in Cranfield.
- PCA offers intuitive validation of embedding quality.
- LambdaRank provides the strongest alignment between training objectives and IR evaluation metrics (MAP, NDCG).
- Combining classical IR with modern learning‑based techniques leads to substantial retrieval improvements.

---

## 📎 References

- Stanford NLP – GloVe (2014): https://nlp.stanford.edu/projects/glove/  
- Springer (2011): https://link.springer.com/chapter/10.1007/978-3-642-18991-3_54  
- ACL Anthology: https://aclanthology.org/C02-1007.pdf  
- Burges (2010): https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf  
- Cranfield Dataset: h-datasets.com/cranfield.html  

---

**Prepared for academic and professional portfolio use.**io use.**

