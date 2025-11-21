import random

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # Store trigram counts: C(w1, w2, w3)
        self.trigram_counts = {}
        # Store bigram counts: C(w1, w2) for smoothing
        self.bigram_counts = {}
        # Vocabulary set for computing V
        self.vocab = set()
        # Special tokens for sentence boundaries
        self.start_token = "<s>"
        self.end_token = "</s>"

    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        # Clean and tokenize: convert to lowercase and split by whitespace
        words = text.lower().split()
        
        if len(words) == 0:
            return
        
        # Add special tokens to mark sentence boundaries
        padded_words = [self.start_token] + words + [self.end_token]
        
        # Build vocabulary (including special tokens)
        self.vocab.update(padded_words)
        
        # Extract and count all trigrams: (w1, w2, w3)
        for i in range(len(padded_words) - 2):
            w1, w2, w3 = padded_words[i], padded_words[i+1], padded_words[i+2]
            
            # Count trigram C(w1, w2, w3)
            trigram = (w1, w2, w3)
            self.trigram_counts[trigram] = self.trigram_counts.get(trigram, 0) + 1
            
            # Count bigram C(w1, w2) for smoothing denominator
            bigram = (w1, w2)
            self.bigram_counts[bigram] = self.bigram_counts.get(bigram, 0) + 1

    def predict_next(self, w1, w2):
        """
        Predicts next word probabilities given two previous words.
        Uses Add-One (Laplace) Smoothing.
        
        Args:
            w1 (str): First context word
            w2 (str): Second context word
            
        Returns:
            dict: {word: probability} for all words in vocabulary
        """
        if not self.vocab:
            return {}
        
        V = len(self.vocab)  # Vocabulary size
        bigram = (w1, w2)
        C_w1w2 = self.bigram_counts.get(bigram, 0)  # Bigram count
        
        probs = {}
        # Apply smoothing formula: P(w3|w1,w2) = (C(w1,w2,w3) + 1) / (C(w1,w2) + V)
        for w3 in self.vocab:
            trigram = (w1, w2, w3)
            C_w1w2w3 = self.trigram_counts.get(trigram, 0)  # Trigram count
            probs[w3] = (C_w1w2w3 + 1) / (C_w1w2 + V)
        
        return probs
    
    def generate(self, max_length=50):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text.

        Returns:
            str: The generated text.
        """
        if not self.vocab:
            return ""
        
        # Start with start token
        w1, w2 = self.start_token, self.start_token
        generated = []
        
        for _ in range(max_length):
            # Get probabilities for next word using smoothing
            probs = self.predict_next(w1, w2)
            
            if not probs:
                break
            
            # Sample next word based on probabilities
            words = list(probs.keys())
            probabilities = list(probs.values())
            w3 = random.choices(words, weights=probabilities)[0]
            
            # Stop if end token is generated
            if w3 == self.end_token:
                break
            
            # Skip start token in output
            if w3 != self.start_token:
                generated.append(w3)
            
            # Shift context window for next prediction
            w1, w2 = w2, w3
        
        return " ".join(generated)
