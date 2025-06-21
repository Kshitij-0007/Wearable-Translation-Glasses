#!/usr/bin/env python3
"""
Create a test image with text for OCR testing
"""

import cv2
import numpy as np
import os

# Create a blank image
img = np.ones((720, 1280, 3), dtype=np.uint8) * 255

# Add some text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "Hello World", (100, 100), font, 2, (0, 0, 0), 3)
cv2.putText(img, "This is a test image", (100, 200), font, 1.5, (0, 0, 0), 2)
cv2.putText(img, "for OCR testing", (100, 250), font, 1.5, (0, 0, 0), 2)
cv2.putText(img, "Welcome to Translation Glasses", (100, 350), font, 1.2, (0, 0, 0), 2)
cv2.putText(img, "Hola Mundo", (100, 450), font, 1.5, (0, 0, 0), 2)
cv2.putText(img, "Bonjour le Monde", (100, 550), font, 1.5, (0, 0, 0), 2)

# Save the image
output_path = os.path.join(os.path.dirname(__file__), "test_image.jpg")
cv2.imwrite(output_path, img)
print(f"Test image created at: {output_path}")
