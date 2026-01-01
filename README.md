### 🔍 Labor Market NLP – Semantic Clustering & O*NET Alignment

A research-driven NLP pipeline for analyzing labor-market data.
This project builds a reproducible workflow to:

* embed job descriptions into a semantic vector space
* cluster occupations by linguistic similarity
* align clusters with **O*NET / SOC taxonomies**
* evaluate clustering + taxonomy consistency

It emphasizes **pipeline design, evaluation rigor, scalability, and reproducibility**, aligning with real-world research expectations.

---

### 🎯 Research Motivation

Modern job postings contain rich but noisy signals about occupations and skills.
This project explores:

* whether embeddings can meaningfully group occupations
* how well unsupervised clusters align with official taxonomies
* how to evaluate systems without labeled ground truth
* what fails — and how to improve it

This is an **exploratory research pipeline**, not a toy demo.

---

### 🧠 Core Objectives

This project demonstrates my ability to:

* design clean end-to-end NLP pipelines
* balance engineering with research thinking
* critically evaluate unsupervised models
* maintain extensibility + reproducibility

---

### 🧰 Technologies

Python · Sentence-Transformers · Scikit-learn
NumPy · Pandas · Matplotlib · Seaborn
PostgreSQL + pgvector (planned)
Config-driven `src/` architecture

---

### ✨ Key Features

✔️ Modular, research-friendly structure
✔️ Embedding + caching pipeline
✔️ K-Means baseline (extendable to HDBSCAN/hierarchical)
✔️ Internal clustering metrics
✔️ O*NET mapping via cosine similarity
✔️ Cluster → SOC reporting
✔️ Quantitative + qualitative evaluation

---

### 🧬 Methodology

1️⃣ Load ~2.4K postings
2️⃣ Generate SBERT embeddings
3️⃣ Cluster (K-Means baseline → future HDBSCAN)
4️⃣ Evaluate: silhouette, Davies–Bouldin
5️⃣ Map to O*NET (embedding similarity, top-k SOC)
6️⃣ Validate: metrics + human review

---

### 📉 Findings & Limitations

* Weak cluster separation (low silhouette)
* Not all clusters align cleanly with taxonomy
* Mapping confidence varies

These are treated as **research signals** guiding next steps.

---

### 🔧 Planned Improvements

🔹 Stronger embeddings
🔹 Dimensionality reduction
🔹 HDBSCAN
🔹 Better taxonomy evaluation
🔹 pgvector integration

---

### 🎮 What Users Can Do

Generate embeddings · cluster jobs · compute metrics · map to O*NET · inspect clusters · reproduce · extend

---

### 🎓 Why It Matters

This project shows **research thinking, engineering discipline, evaluation rigor, and initiative** — the qualities expected from an **NLP / ML Engineer / ML Researcher**.

---

