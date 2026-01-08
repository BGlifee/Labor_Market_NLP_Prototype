###  Labor Market NLP – Semantic Clustering & O*NET Alignment

A research-driven NLP pipeline for analyzing labor-market data.
This project builds a reproducible workflow to:

* embed job descriptions into a semantic vector space
* cluster occupations by linguistic similarity
* align clusters with **O*NET / SOC taxonomies**
* evaluate clustering + taxonomy consistency

---

###  Research Motivation

Modern job postings contain rich but noisy signals about occupations and skills.
This project explores:

* whether embeddings can meaningfully group occupations
* how well unsupervised clusters align with official taxonomies
* how to evaluate systems without labeled ground truth
* what fails — and how to improve it

---

###  Technologies

Python · Sentence-Transformers · Scikit-learn
NumPy · Pandas · Matplotlib · Seaborn
PostgreSQL + pgvector (planned)
Config-driven `src/` architecture

---

###  Key Features

✔️ Modular, research-friendly structure
✔️ Embedding + caching pipeline
✔️ K-Means baseline (extendable to HDBSCAN/hierarchical)
✔️ Internal clustering metrics
✔️ O*NET mapping via cosine similarity
✔️ Cluster → SOC reporting
✔️ Quantitative + qualitative evaluation

---

###  Findings & Limitations

* Weak cluster separation (low silhouette)
* Not all clusters align cleanly with taxonomy
* Mapping confidence varies

---

###  Planned Improvements

🔹 Stronger embeddings
🔹 Dimensionality reduction
🔹 HDBSCAN
🔹 Better taxonomy evaluation
🔹 pgvector integration

---

