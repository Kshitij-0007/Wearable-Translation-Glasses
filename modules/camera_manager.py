"""
Camera Manager Module for Wearable Translation Glasses
Handles camera selection, initialization, and frame capture
"""

import cv2
import time
from typing import Tuple, List, Dict, Any, Optional

class CameraManager:
    def __init__(self, 
                 camera_id: int = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30):
        """
        Initialize the camera manager
        
        Args:
            camera_id: Camera index (0 for built-in, 1+ for external)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.available_cameras = self._detect_cameras()
        
    def _detect_cameras(self) -> List[int]:
        """
        Detect available cameras on the system
        
        Returns:
            List of available camera indices
        """
        available = []
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
    
    def get_available_cameras(self) -> List[int]:
        """
        Get list of available camera indices
        
        Returns:
            List of available camera indices
        """
        return self.available_cameras
    
    def start(self, camera_id: Optional[int] = None) -> bool:
        """
        Start the camera with specified ID or default
        
        Args:
            camera_id: Camera index to use (overrides init value if provided)
            
        Returns:
            True if camera started successfully, False otherwise
        """
        if camera_id is not None:
            self.camera_id = camera_id
            
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
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify settings were applied
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ Camera {self.camera_id} started")
        print(f"   Resolution: {actual_width}x{actual_height} (requested: {self.width}x{self.height})")
        print(f"   FPS: {actual_fps} (requested: {self.fps})")
        
        self.is_running = True
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """
        Read a frame from the camera
        
        Returns:
            Tuple of (success, frame)
        """
        if not self.is_running or self.cap is None:
            return False, None
            
        return self.cap.read()
    
    def release(self) -> None:
        """
        Release the camera resources
        """
        if self.cap is not None:
            self.cap.release()
            self.is_running = False
            print(f"✅ Camera {self.camera_id} released")
