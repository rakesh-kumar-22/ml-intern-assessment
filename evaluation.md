# AI/ML Intern Assignment - Evaluation Document

## Task 1: Trigram Language Model Implementation

### Design Choices and Rationale

#### 1. **Data Structures**
- **Trigram Counts**: Dictionary with tuple keys `(w1, w2, w3)` → count
  - Efficient O(1) lookup for any trigram
  - Memory-efficient for sparse data
- **Bigram Counts**: Dictionary with tuple keys `(w1, w2)` → count
  - Required for Add-One smoothing denominator
- **Vocabulary Set**: Tracks all unique words including special tokens
  - Used to compute V (vocabulary size) for smoothing

#### 2. **Special Tokens**
- `<s>`: Start of sentence token
- `</s>`: End of sentence token
- Follows standard NLP conventions
- Enables model to learn sentence boundaries

#### 3. **Text Preprocessing**
- **Lowercase Normalization**: Reduces vocabulary size, improves generalization
- **Whitespace Tokenization**: Simple but effective for English text
- **Sentence Padding**: Adds `<s>` at start and `</s>` at end

#### 4. **Smoothing Technique: Add-One (Laplace) Smoothing**
- **Formula**: `P(w3|w1,w2) = (C(w1,w2,w3) + 1) / (C(w1,w2) + V)`
- **Why**: Prevents zero probabilities for unseen trigrams
- **Benefit**: Model can handle novel word combinations
- **Trade-off**: Slightly reduces probability of seen trigrams

#### 5. **Next Word Prediction**
- `predict_next(w1, w2)` method returns probability distribution over all vocabulary
- User provides two context words → model predicts next word
- Uses smoothed probabilities for robust predictions
- Returns most likely word based on training data

#### 6. **Text Generation**
- Probabilistic sampling from distribution (not just argmax)
- Creates variety in generated text
- Stops at `</s>` token or max_length
- Shifts context window: (w1, w2) → (w2, w3)

### Implementation Highlights

**Core Algorithm**:
1. Extract all trigrams and bigrams from training text
2. Count frequencies: C(w1,w2,w3) and C(w1,w2)
3. Build vocabulary set V
4. Apply Add-One smoothing for probability estimation
5. Predict next word using smoothed probabilities

**Edge Cases Handled**:
- Empty text → returns empty string
- Unseen bigram context → smoothing ensures all words have non-zero probability
- Short text → boundary tokens provide context

---

## Steps to Run and Test

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
```bash
pytest ml-assignment/tests/test_ngram.py -v
```
**Expected Output**: All 3 tests pass ✅
- `test_fit_and_generate`: Basic functionality
- `test_empty_text`: Edge case handling
- `test_short_text`: Minimal data handling

### 3a. Predict Next Word (Interactive)
```bash
cd ml-assignment/src
python generate.py
```
**What it does**:
- Trains model on corpus
- Prompts user to enter two words
- Predicts and displays the most likely next word
- Shows top 5 predictions with probabilities

**Example Usage**:
```
Enter first word: machine
Enter second word: learning

Predicted next word: is

Top 5 predictions:
1. is              (probability: 0.156)
2. uses            (probability: 0.089)
3. enables         (probability: 0.067)
4. helps           (probability: 0.045)
5. trains          (probability: 0.034)
```

### 3b. Generate Full Text
```bash
cd ml-assignment/src
python generate_text.py
```
**What it does**:
- Generates complete sentences/paragraphs
- Starts from `<s>` tokens
- Continues until `</s>` or max_length reached
- Uses probabilistic sampling for variety

**Example Output**:
```
Enter max length (default 50): 30

Generated Text:
----------------------------------------
machine learning is a fascinating field of artificial
intelligence that enables computers to learn from data
without explicit programming and recognize patterns
----------------------------------------
```

### 4. Download Books from Project Gutenberg (Optional)
```bash
cd ml-assignment/src
python prepare_data.py
```
**What it does**:
- Downloads books from Project Gutenberg
- Removes metadata (headers/footers)
- Saves cleaned text to data folder

**Available books**:
1. Alice's Adventures in Wonderland (ID: 11)
2. Pride and Prejudice (ID: 1342)
3. Frankenstein (ID: 84)
4. A Tale of Two Cities (ID: 98)

After downloading, update the file path in `generate.py`.

### 5. Use Custom Training Data
- Replace `ml-assignment/data/example_corpus.txt` with your text
- Larger corpus = better predictions

---

## Testing Checklist

- [x] Model trains on text corpus
- [x] Trigram and bigram counts extracted correctly
- [x] Vocabulary size computed accurately
- [x] Add-One smoothing applied properly
- [x] predict_next() returns probability distribution
- [x] User can input two words and get next word prediction
- [x] All unit tests pass
- [x] Handles edge cases (empty, short text)
- [x] No zero probabilities (smoothing works)

---

## Results and Performance

**Test Results**: ✅ All tests pass

**Model Capabilities**:
- Predicts next word given two-word context
- Handles unseen word combinations via smoothing
- Generates coherent text sequences
- Probability distribution reflects training data patterns

**Limitations**:
- Only considers last 2 words (Markov assumption)
- Performance depends on training data size
- Simple tokenization (no punctuation handling)
- Add-One smoothing can over-smooth with small datasets

**Potential Improvements**:
- Implement Kneser-Ney smoothing (better than Add-One)
- Add backoff to bigram/unigram models
- Handle punctuation and capitalization
- Support for multiple sentences in generation
- Perplexity evaluation metric

---

## File Structure
```
ml-assignment/
├── data/
│   └── example_corpus.txt    # Training corpus (26 lines)
├── src/
│   ├── ngram_model.py         # TrigramModel class
│   ├── generate.py            # Next word prediction script
│   ├── generate_text.py       # Full text generation script
│   ├── prepare_data.py        # Download books from Gutenberg
│   └── utils.py               # Utility functions
├── tests/
│   └── test_ngram.py          # Unit tests
├── evaluation.md              # Detailed documentation
└── README.md                  # Quick start guide
```

---

## Quick Reference: How to Run

| Task | Command | Description |
|------|---------|-------------|
| **Install** | `pip install -r requirements.txt` | Install dependencies |
| **Test** | `pytest ml-assignment/tests/test_ngram.py -v` | Run unit tests |
| **Predict Next Word** | `cd ml-assignment/src && python generate.py` | Interactive next word prediction |
| **Generate Text** | `cd ml-assignment/src && python generate_text.py` | Generate full text sequences |
| **Download Books** | `cd ml-assignment/src && python prepare_data.py` | Download training data from Gutenberg |

---

## Summary

This implementation provides a complete trigram language model with:
- Proper Add-One (Laplace) smoothing
- Bigram and trigram frequency counting
- Vocabulary tracking for smoothing
- Next word prediction with probability distribution
- Interactive user interface for predictions
- Comprehensive test coverage

The model successfully demonstrates core NLP concepts: n-gram modeling, probability estimation, and smoothing techniques.
