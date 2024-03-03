import re

def detect_link_or_filepath(text):
    # Regular expression pattern to match a URL
    
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    
    # Regular expression pattern to match a file path
    filepath_pattern = r'\b(?:\/\w+)+\.\w{2,4}\b'  # Adjust this pattern as needed
    
    # Check if the text contains a URL or file path
    if re.search(url_pattern, text) or re.search(filepath_pattern, text):
        return 1
    else:
        return 0

# Example usage:
# input_text = input("Enter some text: ")
# result = detect_link_or_filepath(input_text)
# print(result)
