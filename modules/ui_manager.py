"""
UI Manager Module for Wearable Translation Glasses
Handles user interface elements and display
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time

class UIManager:
    def __init__(self, 
                 show_fps: bool = True,
                 show_controls: bool = True,
                 show_debug: bool = False):
        """
        Initialize the UI manager
        
        Args:
            show_fps: Whether to show FPS counter
            show_controls: Whether to show control instructions
            show_debug: Whether to show debug information
        """
        self.show_fps = show_fps
        self.show_controls = show_controls
        self.show_debug = show_debug
        
        # Performance tracking
        self.frame_times = []
        self.last_frame_time = time.time()
        
        # UI colors
        self.colors = {
            'text_bg': (0, 0, 0),
            'text_fg': (255, 255, 255),
            'highlight': (0, 255, 0),
            'warning': (0, 0, 255),
            'box': (0, 255, 0),
            'translation': (255, 255, 0)
        }
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_thickness = 2
        self.font_scale = 0.6
        self.font_scale_large = 0.8
        
    def draw_text_detection(self, 
                           frame: np.ndarray, 
                           detections: List[Tuple],
                           translations: Optional[Dict[str, str]] = None) -> np.ndarray:
        """
        Draw text detection boxes and translations on frame
        
        Args:
            frame: Input image frame
            detections: List of text detections (bbox, text, confidence)
            translations: Optional dictionary mapping original text to translations
            
        Returns:
            Frame with text detections and translations drawn
        """
        result = frame.copy()
        
        for i, (bbox, text, conf) in enumerate(detections):
            # Convert bbox coordinates to integers
            top_left = tuple(map(int, bbox[0]))
            bottom_right = tuple(map(int, bbox[2]))
            
            # Draw bounding box
            cv2.rectangle(result, top_left, bottom_right, self.colors['box'], self.line_thickness)
            
            # Get translated text if available
            translated_text = translations.get(text, text) if translations else text
            
            # Draw original text above the box
            cv2.putText(
                result, f"{text} ({conf:.2f})", 
                (top_left[0], top_left[1] - 10),
                self.font, self.font_scale, self.colors['highlight'], self.line_thickness
            )
            
            # Draw translated text below the box if different from original
            if translations and translated_text != text:
                cv2.putText(
                    result, translated_text, 
                    (top_left[0], bottom_right[1] + 25),
                    self.font, self.font_scale_large, self.colors['translation'], self.line_thickness
                )
        
        return result
    
    def draw_performance_stats(self, 
                              frame: np.ndarray, 
                              stats: Dict[str, Any]) -> np.ndarray:
        """
        Draw performance statistics on frame
        
        Args:
            frame: Input image frame
            stats: Dictionary of performance statistics
            
        Returns:
            Frame with performance statistics drawn
        """
        if not self.show_fps and not self.show_debug:
            return frame
            
        result = frame.copy()
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time) if self.last_frame_time else 0
        self.last_frame_time = current_time
        
        # Keep track of frame times for average FPS
        self.frame_times.append(fps)
        if len(self.frame_times) > 30:  # Keep only last 30 measurements
            self.frame_times.pop(0)
        
        avg_fps = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        
        # Draw FPS counter
        if self.show_fps:
            cv2.putText(
                result, f"FPS: {avg_fps:.1f}", 
                (20, 30),
                self.font, self.font_scale_large, self.colors['text_fg'], self.line_thickness
            )
        
        # Draw additional debug info
        if self.show_debug:
            y_pos = 60
            for key, value in stats.items():
                if isinstance(value, float):
                    text = f"{key}: {value:.3f}"
                else:
                    text = f"{key}: {value}"
                    
                cv2.putText(
                    result, text, 
                    (20, y_pos),
                    self.font, self.font_scale, self.colors['text_fg'], self.line_thickness
                )
                y_pos += 25
        
        return result
    
    def draw_controls(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw control instructions on frame
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with control instructions drawn
        """
        if not self.show_controls:
            return frame
            
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw semi-transparent background for controls
        overlay = result.copy()
        cv2.rectangle(overlay, (0, h-120), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        # Draw control instructions
        controls = [
            "Q: Quit",
            "D: Toggle debug info",
            "F: Toggle FPS display",
            "C: Change camera",
            "S: Source language",
            "T: Target language",
            "R: Reset stabilization"
        ]
        
        y_pos = h - 100
        for control in controls:
            cv2.putText(
                result, control, 
                (20, y_pos),
                self.font, self.font_scale, self.colors['text_fg'], self.line_thickness
            )
            y_pos += 25
        
        return result
    
    def create_settings_view(self, 
                            frame: np.ndarray,
                            settings: Dict[str, Any],
                            options: Dict[str, List[str]]) -> np.ndarray:
        """
        Create a settings view overlay
        
        Args:
            frame: Input image frame
            settings: Current settings values
            options: Available options for each setting
            
        Returns:
            Frame with settings view drawn
        """
        result = frame.copy()
        h, w = result.shape[:2]
        
        # Draw semi-transparent background
        overlay = result.copy()
        cv2.rectangle(overlay, (w//4, h//4), (3*w//4, 3*h//4), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, result, 0.2, 0, result)
        
        # Draw settings title
        cv2.putText(
            result, "Settings", 
            (w//4 + 20, h//4 + 30),
            self.font, self.font_scale_large, self.colors['highlight'], self.line_thickness
        )
        
        # Draw settings and options
        y_pos = h//4 + 70
        for i, (key, value) in enumerate(settings.items()):
            # Setting name
            cv2.putText(
                result, f"{key}:", 
                (w//4 + 20, y_pos),
                self.font, self.font_scale, self.colors['text_fg'], self.line_thickness
            )
            
            # Current value
            cv2.putText(
                result, f"{value}", 
                (w//4 + 150, y_pos),
                self.font, self.font_scale, self.colors['highlight'], self.line_thickness
            )
            
            # Available options
            if key in options:
                option_text = ", ".join(options[key][:5])
                if len(options[key]) > 5:
                    option_text += "..."
                    
                cv2.putText(
                    result, f"Options: {option_text}", 
                    (w//4 + 20, y_pos + 25),
                    self.font, self.font_scale - 0.1, self.colors['text_fg'], 1
                )
                
            y_pos += 60
        
        # Draw close instructions
        cv2.putText(
            result, "Press ESC to close settings", 
            (w//4 + 20, 3*h//4 - 20),
            self.font, self.font_scale, self.colors['warning'], self.line_thickness
        )
        
        return result
