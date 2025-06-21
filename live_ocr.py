#!/usr/bin/env python3
"""
Live OCR with Camera Feed
Detects and displays text in real-time from camera feed
"""

import cv2
import numpy as np
import easyocr
import time
import os
import argparse
from typing import List, Tuple

class LiveOCR:
    def __init__(self, 
                 camera_id: int = 0,
                 target_lang: str = 'en',
                 confidence_threshold: float = 0.4,
                 width: int = 1280,
                 height: int = 720):
        """
        Initialize the Live OCR system
        
        Args:
            camera_id: Camera index to use
            target_lang: Target language for translation
            confidence_threshold: Minimum confidence for text detection
            width: Camera width
            height: Camera height
        """
        self.camera_id = camera_id
        self.target_lang = target_lang
        self.confidence_threshold = confidence_threshold
        self.width = width
        self.height = height
        
        # Fix OpenMP error (Intel conflict)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        # Initialize EasyOCR reader
        print("Initializing OCR engine...")
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("✅ OCR engine initialized")
        
        # For FPS calculation
        self.prev_frame_time = 0
        self.new_frame_time = 0
        
        # For text stabilization
        self.prev_detections = []
        self.stabilization_frames = 3
        self.detection_history = []
        
    def start_camera(self) -> bool:
        """
        Start the camera
        
        Returns:
            True if camera started successfully, False otherwise
        """
        print(f"Starting camera {self.camera_id}...")
        
        # Try to open with DirectShow backend first (Windows)
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        # If failed, try the default backend
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_id)
            
        if not self.cap.isOpened():
            print(f"❌ Failed to open camera {self.camera_id}")
            return False
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Verify settings were applied
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"✅ Camera {self.camera_id} started")
        print(f"   Resolution: {actual_width}x{actual_height}")
        
        return True
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame to enhance text detection
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting conditions
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(thresh, 9, 75, 75)
        
        return filtered
        
    def detect_text(self, frame: np.ndarray) -> List[Tuple]:
        """
        Detect text in the frame
        
        Args:
            frame: Input frame
            
        Returns:
            List of detected text regions with coordinates and confidence
        """
        # Perform OCR on the frame
        raw_results = self.reader.readtext(frame)
        
        # Filter results based on confidence threshold
        filtered_results = []
        for result in raw_results:
            if len(result) >= 3 and result[2] > self.confidence_threshold:
                filtered_results.append(result)
                
        # Update detection history for stabilization
        self.detection_history.append(filtered_results)
        if len(self.detection_history) > self.stabilization_frames:
            self.detection_history.pop(0)
            
        # Stabilize detections
        stable_results = self._stabilize_detections()
        
        return stable_results
        
    def _stabilize_detections(self) -> List[Tuple]:
        """
        Stabilize text detections across multiple frames
        
        Returns:
            Stabilized text detection results
        """
        if not self.detection_history:
            return []
            
        # Dictionary to track text occurrences and positions
        text_occurrences = {}
        
        # Process all detections in history
        for frame_detections in self.detection_history:
            for bbox, text, conf in frame_detections:
                text_key = text.strip().lower()
                
                if text_key not in text_occurrences:
                    text_occurrences[text_key] = {
                        'count': 1,
                        'bbox': [bbox],
                        'conf': [conf],
                        'text': text  # Keep original case
                    }
                else:
                    text_occurrences[text_key]['count'] += 1
                    text_occurrences[text_key]['bbox'].append(bbox)
                    text_occurrences[text_key]['conf'].append(conf)
        
        # Filter and stabilize results
        stable_results = []
        min_occurrences = max(1, self.stabilization_frames // 2)
        
        for text_key, data in text_occurrences.items():
            if data['count'] >= min_occurrences:
                # Use the detection with highest confidence
                best_idx = np.argmax(data['conf'])
                best_bbox = data['bbox'][best_idx]
                best_conf = data['conf'][best_idx]
                original_text = data['text']
                
                stable_results.append((best_bbox, original_text, best_conf))
        
        return stable_results
        
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        """
        Draw text detections on the frame
        
        Args:
            frame: Input frame
            detections: List of text detections
            
        Returns:
            Frame with text detections drawn
        """
        result = frame.copy()
        
        for bbox, text, conf in detections:
            # Convert bbox coordinates to integers
            points = np.array(bbox).astype(np.int32)
            
            # Draw bounding box
            cv2.polylines(result, [points], True, (0, 255, 0), 2)
            
            # Get top-left and bottom-right coordinates
            x_min = min(points[:, 0])
            y_min = min(points[:, 1])
            
            # Draw text with background
            text_with_conf = f"{text} ({conf:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                text_with_conf, font, font_scale, thickness
            )
            
            # Draw background rectangle
            cv2.rectangle(
                result, 
                (x_min, y_min - text_height - 10), 
                (x_min + text_width + 10, y_min), 
                (0, 0, 0), 
                -1
            )
            
            # Draw text
            cv2.putText(
                result, 
                text_with_conf, 
                (x_min + 5, y_min - 5), 
                font, 
                font_scale, 
                (0, 255, 0), 
                thickness
            )
            
        return result
        
    def draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw FPS counter on the frame
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with FPS counter drawn
        """
        # Calculate FPS
        self.new_frame_time = time.time()
        fps = 1 / (self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
        self.prev_frame_time = self.new_frame_time
        
        # Draw FPS counter
        cv2.putText(
            frame, 
            f"FPS: {fps:.1f}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )
        
        return frame
        
    def run(self):
        """
        Run the Live OCR system
        """
        # Start camera
        if not self.start_camera():
            print("❌ Failed to start camera. Exiting.")
            return
            
        print("✅ Live OCR started")
        print("   Press 'q' to quit")
        print("   Press 's' to save current frame")
        
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
                
            # Process frame for better OCR
            processed = self.process_frame(frame)
            
            # Detect text
            detections = self.detect_text(processed)
            
            # Draw detections on original frame
            result = self.draw_detections(frame, detections)
            
            # Draw FPS counter
            result = self.draw_fps(result)
            
            # Display result
            cv2.imshow("Live OCR", result)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                output_path = f"ocr_capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(output_path, result)
                print(f"✅ Frame saved to {output_path}")
                
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Live OCR stopped")
        
    def run_with_test_image(self, image_path: str):
        """
        Run with a test image when camera is not available
        
        Args:
            image_path: Path to test image
        """
        print(f"Running with test image: {image_path}")
        
        # Load test image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Failed to load test image from {image_path}")
            return
            
        # Process frame for better OCR
        processed = self.process_frame(frame)
        
        # Detect text
        detections = self.detect_text(processed)
        print(f"Found {len(detections)} text regions:")
        for i, (_, text, conf) in enumerate(detections):
            print(f"  {i+1}. '{text}' (Confidence: {conf:.2f})")
        
        # Draw detections on original frame
        result = self.draw_detections(frame, detections)
        
        # Save the result
        output_path = "ocr_result.jpg"
        cv2.imwrite(output_path, result)
        print(f"✅ Result saved to {output_path}")
        
        # Try to display the result
        try:
            cv2.imshow("Live OCR (Test Image)", result)
            print("Press any key to exit")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Note: Could not display result window: {e}")

def main():
    """
    Main entry point
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Live OCR with Camera Feed')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--target', type=str, default='en', help='Target language code (default: en)')
    parser.add_argument('--width', type=int, default=1280, help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Camera height (default: 720)')
    parser.add_argument('--test', action='store_true', help='Run with test image')
    parser.add_argument('--confidence', type=float, default=0.4, help='Confidence threshold (default: 0.4)')
    args = parser.parse_args()
    
    # Create Live OCR system
    ocr = LiveOCR(
        camera_id=args.camera,
        target_lang=args.target,
        confidence_threshold=args.confidence,
        width=args.width,
        height=args.height
    )
    
    # Run the system
    if args.test:
        test_image_path = os.path.join(os.path.dirname(__file__), "test_images", "test_image.jpg")
        ocr.run_with_test_image(test_image_path)
    else:
        try:
            ocr.run()
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Falling back to test image...")
            test_image_path = os.path.join(os.path.dirname(__file__), "test_images", "test_image.jpg")
            if os.path.exists(test_image_path):
                ocr.run_with_test_image(test_image_path)
            else:
                print("❌ No test image available")

if __name__ == "__main__":
    main()
