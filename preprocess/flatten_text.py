"""
Text Flattener - Remove line breaks for clean JSON formatting
"""

def flatten_text(input_file="input_text.txt", output_file="output_text.txt"):
    """Read text file and remove all line breaks."""
    
    # Read the file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Remove all newlines and replace with single space
    flattened = ' '.join(text.split())
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(flattened)
    
    print(f"✓ Flattened text written to: {output_file}")
    print(f"  Original length: {len(text)} chars")
    print(f"  Flattened length: {len(flattened)} chars")
    print(f"\nFirst 100 chars:")
    print(f"  {flattened[:100]}...")


if __name__ == "__main__":
    flatten_text()