# Ghost Job Detector — Semantic Detection of Recurrent Job Postings

Questions
How effectively can lexical, semantic, and hybrid retrieval identify recurrent job postings across time?
What temporal patterns characterize recurrent technology job postings?

Purpose
Detecting Persistent and Semantically Duplicate Job Postings Using Hybrid Information Retrieval.

## 1. Data Source
1. Open-Apply Jobs: https://huggingface.co/datasets/edwarddgao/open-apply-jobs
2. LinkedIn Job Postings: 2023–2024 https://www.kaggle.com/datasets/arshkon/linkedin-job-postings/data
3. historical dataset: https://www.kaggle.com/datasets/madhab/jobposts

## 2. Expected Architecture
```
              Open-Apply
                 │
        Historical snapshots
                 │
        ┌────────┴────────┐
        │                 │
 Exact lifecycle     New posting
   tracking               │
                          ▼
                  Candidate retrieval
                          │
              ┌───────────┼───────────┐
              ↓           ↓           ↓
            BM25        Vector      Hybrid
              │           │           │
              └───────────┼───────────┘
                          ↓
                     Evaluation
                          │
                          ▼
                 Recurrent postings
                          │
                          ▼
                  Temporal patterns
```
