
# 📝 Vietnamese Student Feedback Classification (Sentiment & Topic)

This project presents a machine learning-based approach for analyzing student feedback in Vietnamese. The objective is to automatically classify both the **sentiment** and the **topic** of each sentence using traditional NLP techniques and efficient classifiers.

---

## 📌 Project Overview

- **Task**: Dual-label classification:
  - **Sentiment**: Positive, Neutral, Negative
  - **Topic**: Lecturer, Curriculum, Facility, Others
- **Dataset**: UIT-VSFC (Vietnamese Students’ Feedback Corpus)
- **Language**: Vietnamese
- **Approach**: TF-IDF + Passive-Aggressive Classifier

---

## 🗃️ Dataset: UIT-VSFC

- Collected from university student feedback in Vietnam.
- Includes ~5000 samples with gold standard annotations.
- Labeled by experts for both sentiment and topic.
- Format:
  - `sents.txt`: Sentences
  - `sentiments.txt`: Sentiment labels
  - `topics.txt`: Topic labels
- [UIT-VSFC Corpus Reference](https://uit-nlp.github.io/datasets/uit-vsfc.html)

---

## ⚙️ Methodology

### 1. Preprocessing

- Remove special characters and normalize casing/punctuation.
- Convert text into numerical features using:
  - `TfidfVectorizer` with `ngram_range=(1, 4)`
  - `TfidfTransformer` with `sublinear_tf=True` (log-scale weighting)

### 2. Classification Model

- **Passive-Aggressive Classifier (PA)**: A variation of the Perceptron algorithm.
- Reasons for choosing PA:
  - Faster training than neural networks.
  - Effective on sparse datasets like TF-IDF.
  - Configured with:
    - `C=1` (max step size)
    - `loss="hinge"`
    - `class_weight="balanced"`
    - `n_jobs=-1` (multi-threaded)

---

## 📊 Results on Test Set

- Sentiment and topic classifiers both show competitive performance compared to prior studies using BiLSTM or MaxEnt.
- Key evaluation metrics:
  - Precision
  - Recall
  - F1-score (per class and weighted average)

> Note: Label noise in some annotated data may affect absolute performance.

---

## ✅ Pros and Cons

### ✅ Advantages
- Simple and fast training process.
- Interpretable results and lightweight model.
- Language-agnostic — applicable to other languages.

### ⚠️ Limitations
- TF-IDF does not capture word semantics or order.
- Sensitive to corpus and document length.
- Cannot differentiate between semantically similar phrases with different structures.

---

## 💡 Future Improvements

- Use pretrained embeddings (e.g., Word2Vec, FastText).
- Fine-tune transformers (PhoBERT, BERTvi).
- Combine syntactic and semantic features for better generalization.

---

## 👥 Team Members

- Nguyễn Thành Vinh — 21020710  
- Phạm Thành Vinh — 20021477

University of Engineering and Technology – VNU Hanoi

---

## 📚 References

- Pedregosa et al., “Scikit-learn: Machine Learning in Python”, JMLR 12, 2011  
- UIT-VSFC: Vietnamese Students’ Feedback Corpus  
- [GitHub: Vietnamese Sentiment Analysis - tuanpham1989](https://github.com/tuanpham1989/sentiment_analysis_nal)  
- Nguyen et al., “Deep Learning vs Traditional Classifiers on Vietnamese Feedback”, NICS 2018  
- Nguyen et al., “UIT-VSFC Corpus for Sentiment Analysis”, KSE 2018

