import re

def validate_plate(text):
    """
    Validates the OCR text to see if it looks like a license plate.
    Returns the cleaned string if valid, else None.
    """
    # Remove any non-alphanumeric characters
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Standard license plate is typically between 5 and 8 characters.
    if 5 <= len(cleaned) <= 8:
        # Require at least one letter and at least one number
        if re.search(r'[A-Z]', cleaned) and re.search(r'[0-9]', cleaned):
            return cleaned
            
    return None
