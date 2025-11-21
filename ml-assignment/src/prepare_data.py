import requests
import re

def download_gutenberg_book(book_id, output_file):
    """
    Downloads a book from Project Gutenberg and saves it to a file.
    
    Args:
        book_id (int): Project Gutenberg book ID
        output_file (str): Path to save the downloaded text
    """
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    
    print(f"Downloading book {book_id} from Project Gutenberg...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Remove Gutenberg header and footer
        text = response.text
        text = strip_gutenberg_metadata(text)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Successfully downloaded and saved to {output_file}")
        print(f"Text length: {len(text)} characters")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading book: {e}")
        print("Trying alternative URL format...")
        
        # Try alternative URL
        alt_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        try:
            response = requests.get(alt_url)
            response.raise_for_status()
            text = strip_gutenberg_metadata(response.text)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            print(f"Successfully downloaded and saved to {output_file}")
            print(f"Text length: {len(text)} characters")
        except:
            print("Failed to download. Please download manually from gutenberg.org")

def strip_gutenberg_metadata(text):
    """
    Removes Project Gutenberg header and footer metadata.
    
    Args:
        text (str): Raw text from Gutenberg
        
    Returns:
        str: Cleaned text
    """
    # Remove header
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT"
    ]
    
    for marker in start_markers:
        if marker in text:
            text = text.split(marker, 1)[-1]
            break
    
    # Remove footer
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG"
    ]
    
    for marker in end_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    
    return text.strip()

def main():
    """
    Download recommended books for training the trigram model.
    """
    print("=" * 60)
    print("Project Gutenberg Book Downloader")
    print("=" * 60)
    
    # Recommended books with their IDs
    books = {
        "1": ("Alice's Adventures in Wonderland", 11),
        "2": ("Pride and Prejudice", 1342),
        "3": ("Frankenstein", 84),
        "4": ("A Tale of Two Cities", 98)
    }
    
    print("\nRecommended books:")
    for key, (title, book_id) in books.items():
        print(f"{key}. {title} (ID: {book_id})")
    
    choice = input("\nSelect a book (1-4) or enter custom book ID: ").strip()
    
    if choice in books:
        title, book_id = books[choice]
        output_file = f"../data/{title.lower().replace(' ', '_').replace(\"'\", '')}.txt"
    else:
        try:
            book_id = int(choice)
            output_file = f"../data/book_{book_id}.txt"
        except ValueError:
            print("Invalid input. Exiting.")
            return
    
    download_gutenberg_book(book_id, output_file)
    
    print("\n" + "=" * 60)
    print("To use this corpus, update the path in generate.py:")
    print(f'with open("{output_file}", "r") as f:')
    print("=" * 60)

if __name__ == "__main__":
    main()
