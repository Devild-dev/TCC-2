import cv2
import pytesseract

def run_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("⚠️ Failed to load image for OCR.")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    config = r'--oem 3 --psm 6'
    threshold_modes = ["THRESH_BINARY_INV", "THRESH_BINARY", "THRESH_TRUNC"]
    ocr_results = []

    for mode in threshold_modes:
        threshold_type = getattr(cv2, mode)
        for thresh_value in range(0, 250, 30):
            _, thresh = cv2.threshold(gray, thresh_value, 255, threshold_type)
            text = pytesseract.image_to_string(thresh, config=config)
            words = [w.lower() for w in text.split() if w.strip()]
            ocr_results.append(words)

    return ocr_results
