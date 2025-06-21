"""
Image Processing Module for Wearable Translation Glasses
Handles image enhancement, stabilization, and tracking
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class ImageProcessor:
    def __init__(self, 
                 stabilization_strength: float = 0.8,
                 enhance_contrast: bool = True,
                 denoise_strength: int = 5):
        """
        Initialize the image processor with specified settings
        
        Args:
            stabilization_strength: Strength of the stabilization (0-1)
            enhance_contrast: Whether to enhance image contrast
            denoise_strength: Strength of denoising (higher = more smoothing)
        """
        self.stabilization_strength = stabilization_strength
        self.enhance_contrast = enhance_contrast
        self.denoise_strength = denoise_strength
        
        # For image stabilization
        self.prev_gray = None
        self.prev_points = None
        self.transform_matrix = np.eye(2, 3, dtype=np.float32)
        self.smoothed_transform = np.eye(2, 3, dtype=np.float32)
        
        # For motion blur detection
        self.prev_frame = None
        self.motion_threshold = 30
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame with enhancement and stabilization
        
        Args:
            frame: Input image frame
            
        Returns:
            Processed image frame
        """
        # Check if frame is valid
        if frame is None or frame.size == 0:
            return frame
        
        # Make a copy to avoid modifying the original
        processed = frame.copy()
        
        # Apply image stabilization
        if self.prev_gray is not None:
            processed = self._stabilize_frame(processed)
        
        # Enhance image quality
        processed = self._enhance_image(processed)
        
        # Update previous frame for next iteration
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        self.prev_frame = frame.copy()
        
        return processed
    
    def _stabilize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Stabilize frame to reduce camera shake
        
        Args:
            frame: Input image frame
            
        Returns:
            Stabilized image frame
        """
        # Convert current frame to grayscale
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect good features to track
        if self.prev_points is None:
            self.prev_points = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=200, qualityLevel=0.01, 
                minDistance=30, blockSize=3
            )
            
            if self.prev_points is None or len(self.prev_points) < 10:
                # Not enough points to track
                return frame
        
        # Calculate optical flow
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_points, None
        )
        
        # Filter only valid points
        idx = np.where(status == 1)[0]
        if len(idx) < 10:
            # Not enough points tracked successfully
            return frame
            
        prev_points = self.prev_points[idx].reshape(-1, 2)
        curr_points = curr_points[idx].reshape(-1, 2)
        
        # Estimate rigid transformation
        transform = cv2.estimateAffinePartial2D(prev_points, curr_points)[0]
        
        if transform is None:
            return frame
            
        # Update transformation matrix
        self.transform_matrix = transform
        
        # Smooth the transformation for more stable results
        self.smoothed_transform = (
            self.stabilization_strength * self.smoothed_transform + 
            (1 - self.stabilization_strength) * self.transform_matrix
        )
        
        # Apply smoothed transformation
        h, w = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame, self.smoothed_transform, (w, h), 
            flags=cv2.INTER_LINEAR
        )
        
        # Update points for next frame
        self.prev_points = curr_points.reshape(-1, 1, 2)
        
        return stabilized
    
    def _enhance_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance image quality for better OCR
        
        Args:
            frame: Input image frame
            
        Returns:
            Enhanced image frame
        """
        # Check if motion blur is present
        is_blurry = self._detect_motion_blur(frame)
        
        # Apply denoising (stronger if motion blur detected)
        denoise_strength = self.denoise_strength * 2 if is_blurry else self.denoise_strength
        denoised = cv2.fastNlMeansDenoisingColored(
            frame, None, denoise_strength, denoise_strength, 7, 21
        )
        
        if self.enhance_contrast:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            
            # Merge channels and convert back to BGR
            enhanced_lab = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
        
        return denoised
    
    def _detect_motion_blur(self, frame: np.ndarray) -> bool:
        """
        Detect if the current frame has motion blur
        
        Args:
            frame: Input image frame
            
        Returns:
            True if motion blur is detected, False otherwise
        """
        if self.prev_frame is None:
            return False
            
        # Calculate frame difference
        diff = cv2.absdiff(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        )
        
        # Calculate Laplacian variance (blurry images have lower variance)
        lap_var = cv2.Laplacian(frame, cv2.CV_64F).var()
        
        # High difference and low Laplacian variance indicates motion blur
        avg_diff = np.mean(diff)
        return avg_diff > self.motion_threshold and lap_var < 100
