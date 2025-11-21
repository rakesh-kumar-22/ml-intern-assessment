from ngram_model import TrigramModel

def main():
    # Create and train the model
    model = TrigramModel()
    
    with open("../data/example_corpus.txt", "r") as f:
        text = f.read()
    model.fit(text)
    
    print("Trigram Model - Text Generation")
    print("=" * 40)
    
    # Generate text
    try:
        user_input = input("\nEnter max length (default 50): ").strip()
        max_length = int(user_input) if user_input else 50
    except ValueError:
        print("Invalid input. Using default length of 50.")
        max_length = 50
    
    print("\nGenerating text...\n")
    generated_text = model.generate(max_length=max_length)
    
    print("Generated Text:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)

if __name__ == "__main__":
    main()
