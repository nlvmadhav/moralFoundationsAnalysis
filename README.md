# Moral Foundation Analysis of Twitter Tweets (Indian Context)

## Project Overview
This research-driven project aims to analyze Twitter discourse in the **Indian socio-political context** through the framework of **Moral Foundation Theory (MFT)**. The objective is to classify tweets based on underlying moral values and conduct **sentiment analysis** 

### **Data Preprocessing Steps:**
1. **Text Cleaning:** Removal of special characters, stopwords, and irrelevant symbols.
2. **Tokenization & Lemmatization:** Implemented using **SpaCy** for efficient processing.
3. **Feature Extraction:** Utilized **TF-IDF, Word2Vec, and BERT embeddings** to encode text for classification.

## Moral Foundation Classification
Moral Foundations are identified using the **Moral Foundation Dictionary (MFD)**, a lexicon-based approach for text classification, and enhanced using machine learning techniques.

- **Supervised Learning Approaches:**
  - **Deep Learning Architectures:** BiLSTM, Transformer-based models (BERT).
  - **Fine-tuned Moral Foundation Models:** Leveraging BERT-based models trained on domain-specific moral corpora.

- **Lexicon-Based Classification:**
  - Word-matching techniques with MFD.
  - Semantic similarity methods using **Word2Vec and Sentence-BERT (SBERT)**.

## Sentiment Analysis Methodology
- **Sentiment Classification:** Compared
  - **LSTM** for sentiment detection.
  - **BERT-based sentiment classification** with transformer fine-tuning.
- **Evaluation Metrics:**
  - Precision, Recall, F1-Score for performance validation.
  - **K-Fold Cross-Validation** to ensure model robustness.
    
## Moral Foundation Analysis
- **Sentiment Classification:** Compared
  - **LSTM** for Moral Foundation classifications.
  - **BERT-based sentiment classification** with transformer fine-tuning.
- **Evaluation Metrics:**
  - Precision, Recall, F1-Score for performance validation.
  - **K-Fold Cross-Validation** to ensure model robustness.

## Technology Stack
- **Programming Language:** Python (Pandas, NumPy, Matplotlib, Seaborn)
- **Natural Language Processing (NLP):**  Transformers (Hugging Face)
- **Machine Learning & Deep Learning:**
  - Pytorch (LSTMs, BiLSTMs, Transformers)
- **Feature Embeddings:**
  - TF-IDF, BERT (Hugging Face Transformers)

## Implementation & Usage
1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/moralFoundations.git
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the analysis pipeline:**
   ```sh
   python analyze_tweets.py
   ```

## Research Contributions & Future Work
- **Enhancing model interpretability:** Leveraging SHAP & LIME for explainability.
- **Expanding dataset:** Inclusion of more diverse socio-political themes.
- **Deploying a real-time dashboard:** Web-based monitoring using Flask/Django.

## Contributions & Citation
Researchers and developers are encouraged to contribute. Fork the repository, create feature branches, and submit pull requests.

For citation:
```bibtex
@article{moralFoundationAndSentimentAnalysis2025,
  author    = {NLV Madhav},
  title     = {Moral Foundation Analysis of Twitter Tweets in the Indian Context},
  year      = {2025},
  url       = {https://github.com/your-username/moralFoundations}
}
```

## License
This project is released under the **MIT License**.
