import cv2
import pytesseract

def extract_text(aligned_plate):
    """
    Extracts text from the aligned plate using Tesseract OCR.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(aligned_plate, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to get a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Pad the image to give tesseract a better chance
    thresh = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255,255,255])
    
    # Configuration for Tesseract: 
    # --psm 7: Treat the image as a single text line
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    text = pytesseract.image_to_string(thresh, config=config)
    
    return text.strip()
