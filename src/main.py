import cv2
import os
import csv
from datetime import datetime
from collections import defaultdict

from detect import detect_plate
from align import align_plate
from ocr import extract_text
from validate import validate_plate

def main():
    # Setup directories relative to this script
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    SCREENSHOTS_DIR = os.path.join(BASE_DIR, 'screenshots')
    
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    
    csv_file = os.path.join(DATA_DIR, 'plates.csv')
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Plate_Number'])
            
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    plate_history = defaultdict(int)
    confirmed_plates = set()
    screenshots_saved = False
    
    # Confirmation threshold (number of identical observations needed)
    CONFIRMATION_THRESHOLD = 5
    
    print("Starting ANPR Pipeline. Point your webcam at a license plate.")
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        
        # 1. Detect
        contour = detect_plate(frame)
        
        if contour is not None:
            # Draw contour on display frame (green box)
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
            
            # 2. Align
            aligned = align_plate(frame, contour)
            
            # 3. OCR
            text = extract_text(aligned)
            
            # 4. Validate
            valid_plate = validate_plate(text)
            
            if valid_plate:
                # Output text to screen
                cv2.putText(display_frame, valid_plate, 
                            (max(0, contour[0][0][0]), max(0, contour[0][0][1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                # 5. Confirm after multiple observations
                if valid_plate not in confirmed_plates:
                    plate_history[valid_plate] += 1
                    
                    if plate_history[valid_plate] >= CONFIRMATION_THRESHOLD:
                        confirmed_plates.add(valid_plate)
                        print(f"\nCONFIRMED PLATE: {valid_plate}")
                        
                        # Save to CSV
                        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), valid_plate])
                            
                        # Save screenshots on first confirmation
                        if not screenshots_saved:
                            cv2.imwrite(os.path.join(SCREENSHOTS_DIR, 'detection.png'), display_frame)
                            cv2.imwrite(os.path.join(SCREENSHOTS_DIR, 'alignment.png'), aligned)
                            
                            # Create an OCR debug image
                            ocr_debug = aligned.copy()
                            cv2.putText(ocr_debug, valid_plate, (10, int(ocr_debug.shape[0] / 2)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.imwrite(os.path.join(SCREENSHOTS_DIR, 'ocr.png'), ocr_debug)
                            
                            screenshots_saved = True
                            print(f"Screenshots saved to {SCREENSHOTS_DIR}")
                else:
                    cv2.putText(display_frame, "SAVED", 
                                (max(0, contour[0][0][0]), min(display_frame.shape[0], contour[0][0][1] + 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            
            # Show aligned plate in a separate small window
            cv2.imshow('Aligned Plate', aligned)
            
        cv2.imshow('ANPR Camera', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
