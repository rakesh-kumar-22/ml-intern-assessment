# Trigram Language Model

This directory contains the core assignment files for the Trigram Language Model.

## How to Run 

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
pytest ml-assignment/tests/test_ngram.py -v
```
Expected: All 3 tests pass ✅

### 3a. Predict Next Word (Interactive)
```bash
cd ml-assignment/src
python generate.py
```

**Usage**:
- Enter two words when prompted
- Model predicts the most likely next word
- Shows top 5 predictions with probabilities

**Example**:
```
Enter first word: machine
Enter second word: learning

Predicted next word: is
Probability: 0.1234

Top 5 predictions:
1. is              (probability: 0.1234)
2. uses            (probability: 0.0891)
3. enables         (probability: 0.0678)
...
```

### 3b. Generate Full Text
```bash
cd ml-assignment/src
python generate_text.py
```

**Usage**:
- Enter desired text length
- Model generates complete text
- Uses probabilistic sampling

**Example**:
```
Enter max length (default 50): 30

Generated Text:
----------------------------------------
machine learning is a fascinating field...
----------------------------------------
```

### 4. Download Books from Project Gutenberg (Optional)
```bash
cd ml-assignment/src
python prepare_data.py
```
Select from recommended books:
- Alice's Adventures in Wonderland
- Pride and Prejudice
- Frankenstein
- A Tale of Two Cities

Then update the path in `generate.py` to use the downloaded book.

### 5. Use Custom Corpus
- Replace `ml-assignment/data/example_corpus.txt` with your text
- Larger corpus = better predictions

## Design Choices

### Trigram Model with Add-One Smoothing
- Uses formula: `P(w3|w1,w2) = (C(w1,w2,w3) + 1) / (C(w1,w2) + V)`
- Tracks trigram counts, bigram counts, and vocabulary
- Prevents zero probabilities for unseen word combinations
- `predict_next(w1, w2)` returns probability distribution

See `evaluation.md` for detailed design decisions and rationale.
