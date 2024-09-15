# ocr_helper.py

import easyocr
from PIL import Image
import cv2
import numpy as np
import io

def preprocess_image(image_bytes):
    """
    Preprocesses the image to enhance OCR accuracy.
    """
    try:
        # Open the image using PIL
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_np = np.array(image)

        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11, 
            2
        )

        # Perform morphological operations to enhance text regions
        kernel = np.ones((3,3), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        return processed

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        # If preprocessing fails, return the original grayscale image
        try:
            gray = cv2.cvtColor(np.array(Image.open(io.BytesIO(image_bytes)).convert('RGB')), cv2.COLOR_RGB2GRAY)
            return gray
        except Exception as inner_e:
            print(f"Error during fallback preprocessing: {inner_e}")
            return None  # Return None if all preprocessing steps fail

def initialize_reader(languages=['en'], use_gpu=False):
    """
    Initializes the EasyOCR Reader.
    """
    try:
        reader = easyocr.Reader(languages, gpu=use_gpu)
        return reader
    except Exception as e:
        print(f"Error initializing EasyOCR Reader: {e}")
        raise e

def perform_ocr(reader, image_array):
    """
    Performs OCR on the given image using the provided EasyOCR Reader.
    """
    try:
        results = reader.readtext(image_array, detail=0, paragraph=True)
        extracted_text = "\n".join(results)
        return extracted_text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def ocr_helper(uploaded_file, languages=['en'], preprocess=True, use_gpu=False):
    """
    Extracts text from the given image uploaded via Streamlit.
    """
    try:
        # Read the uploaded file as bytes
        image_bytes = uploaded_file.read()

        # Optionally preprocess the image
        if preprocess:
            processed_image = preprocess_image(image_bytes)
            if processed_image is None:
                print(f"Preprocessing failed for file: {uploaded_file.name}. Using original image.")
                # Fallback: Convert image bytes to NumPy array without preprocessing
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                processed_image = np.array(image)
        else:
            # Convert image bytes to NumPy array without preprocessing
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            processed_image = np.array(image)

        # Initialize the EasyOCR reader
        reader = initialize_reader(languages, use_gpu)

        # Perform OCR
        extracted_text = perform_ocr(reader, processed_image)

        return extracted_text

    except Exception as e:
        print(f"Error in ocr_helper: {e}")
        return ""
