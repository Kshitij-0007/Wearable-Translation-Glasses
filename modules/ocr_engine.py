"""
OCR Engine Module for Wearable Translation Glasses
Handles text detection and recognition with enhanced stability and accuracy
"""

import os
import cv2
import easyocr
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time

class OCREngine:
    def __init__(self, 
                 languages: List[str] = ['en'], 
                 gpu: bool = False,
                 confidence_threshold: float = 0.4,
                 stabilization_frames: int = 5):
        """
        Initialize the OCR engine with specified languages and settings
        
        Args:
            languages: List of language codes to detect
            gpu: Whether to use GPU acceleration
            confidence_threshold: Minimum confidence score to accept detection
            stabilization_frames: Number of frames to use for stabilization
        """
        # Fix OpenMP error (Intel conflict)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(languages, gpu=gpu)
        self.confidence_threshold = confidence_threshold
        
        # Text stabilization parameters
        self.stabilization_frames = stabilization_frames
        self.detection_history = []
        self.stable_results = []
        
        # Performance tracking
        self.processing_times = []
        self.last_frame_time = time.time()
    
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance image for better OCR performance
        
        Args:
            frame: Input image frame
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding to handle varying lighting conditions
        # This helps with text detection in different lighting scenarios
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to remove noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(opening, 9, 75, 75)
        
        return filtered
    
    def detect_text(self, frame: np.ndarray) -> List[Tuple]:
        """
        Detect text in the given frame with enhanced accuracy
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detected text regions with coordinates and confidence
        """
        start_time = time.time()
        
        # Preprocess the image for better OCR
        processed_frame = self.preprocess_image(frame)
        
        # Perform OCR on the processed frame
        raw_results = self.reader.readtext(processed_frame, paragraph=True)
        
        # Filter results based on confidence threshold
        filtered_results = []
        for result in raw_results:
            if len(result) >= 3 and result[2] > self.confidence_threshold:
                filtered_results.append(result)
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:  # Keep only last 30 measurements
            self.processing_times.pop(0)
        
        # Update detection history for stabilization
        self.update_detection_history(filtered_results)
        
        return self.stable_results
    
    def update_detection_history(self, current_detections: List[Tuple]) -> None:
        """
        Update detection history and stabilize results across frames
        
        Args:
            current_detections: Current frame's text detections
        """
        # Add current detections to history
        self.detection_history.append(current_detections)
        
        # Keep only the last N frames
        if len(self.detection_history) > self.stabilization_frames:
            self.detection_history.pop(0)
        
        # Stabilize detections across frames
        self.stable_results = self._stabilize_detections()
    
    def _stabilize_detections(self) -> List[Tuple]:
        """
        Stabilize text detections across multiple frames to handle camera movement
        
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
                        'conf': [conf]
                    }
                else:
                    text_occurrences[text_key]['count'] += 1
                    text_occurrences[text_key]['bbox'].append(bbox)
                    text_occurrences[text_key]['conf'].append(conf)
        
        # Filter and stabilize results
        stable_results = []
        min_occurrences = max(1, self.stabilization_frames // 2)
        
        for text, data in text_occurrences.items():
            if data['count'] >= min_occurrences:
                # Use the detection with highest confidence
                best_idx = np.argmax(data['conf'])
                best_bbox = data['bbox'][best_idx]
                best_conf = data['conf'][best_idx]
                
                stable_results.append((best_bbox, text, best_conf))
        
        return stable_results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics of the OCR engine
        
        Returns:
            Dictionary with performance metrics
        """
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0
        self.last_frame_time = current_time
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'fps': fps,
            'avg_processing_time': avg_processing_time,
            'confidence_threshold': self.confidence_threshold
        }
