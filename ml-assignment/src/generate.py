from ngram_model import TrigramModel

def main():
    # Create and train the model
    model = TrigramModel()
    
    with open("../data/example_corpus.txt", "r") as f:
        text = f.read()
    model.fit(text)
    
    print("Trigram Model - Next Word Prediction")
    print("=" * 40)
    
    # Get user input for two words
    w1 = input("\nEnter first word: ").lower().strip()
    w2 = input("Enter second word: ").lower().strip()
    
    # Get probability distribution for next word
    probs = model.predict_next(w1, w2)
    
    if not probs:
        print("\nModel not trained. Please train the model first.")
        return
    
    # Get top prediction
    best_word = max(probs, key=probs.get)
    
    print(f"\nPredicted next word: {best_word}")
    print(f"Probability: {probs[best_word]:.4f}")
    
    # Show top 5 predictions
    print("\nTop 5 predictions:")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (word, prob) in enumerate(sorted_probs, 1):
        print(f"{i}. {word:15s} (probability: {prob:.4f})")

if __name__ == "__main__":
    main()
