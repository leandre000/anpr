import cv2
import imutils

def detect_plate(frame):
    """
    Detects a license plate in the given frame.
    Returns the contour of the plate if found, else None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Edge detection
    edged = cv2.Canny(bfilter, 30, 200)
    
    # Find contours
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    
    # Sort out top 10 contours based on area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    location = None
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        
        # If our approximated contour has four points, we can assume we found our plate
        if len(approx) == 4:
            location = approx
            break
            
    return location
