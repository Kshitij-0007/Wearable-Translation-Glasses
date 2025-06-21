#!/usr/bin/env python3
"""
Wearable Translation Glasses with AR Integration
Main application file

This application provides real-time OCR and translation with stabilization
to simulate AR glasses experience. It uses modular components for camera
management, image processing, OCR, translation, and UI display.

Usage:
    python translation_glasses.py [--camera CAMERA_ID] [--source SOURCE_LANG] [--target TARGET_LANG]
    python translation_glasses.py --test  # Run in test mode with a static image
"""

import cv2
import numpy as np
import argparse
import time
import threading
from queue import Queue
import os

# Import custom modules
from modules.camera_manager import CameraManager
from modules.image_processor import ImageProcessor
from modules.ocr_engine import OCREngine
from modules.translator import Translator
from modules.ui_manager import UIManager

class TranslationGlasses:
    def __init__(self, 
                 camera_id: int = 0,
                 source_lang: str = 'auto',
                 target_lang: str = 'en',
                 width: int = 1280,
                 height: int = 720):
        """
        Initialize the Translation Glasses application
        
        Args:
            camera_id: Camera index to use
            source_lang: Source language code
            target_lang: Target language code
            width: Display width
            height: Display height
        """
        # Initialize components
        self.camera = CameraManager(camera_id, width, height)
        self.image_processor = ImageProcessor(stabilization_strength=0.8)
        self.ocr_engine = OCREngine(languages=['en'], stabilization_frames=5)
        self.translator = Translator(source_lang, target_lang)
        self.ui = UIManager(show_fps=True, show_controls=True)
        
        # Application state
        self.running = False
        self.show_settings = False
        self.current_settings = {
            'Camera': camera_id,
            'Source Language': source_lang,
            'Target Language': target_lang,
            'Stabilization': 'On'
        }
        
        # Processing queues and threads
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.processing_thread = None
        
        # Performance tracking
        self.stats = {
            'ocr_time': 0.0,
            'translation_time': 0.0,
            'processing_time': 0.0,
            'detections': 0
        }
        
    def start(self):
        """
        Start the application
        """
        # Start camera
        if not self.camera.start():
            print("❌ Failed to start camera.")
            print("Falling back to test mode...")
            self._run_with_test_image()
            return
            
        # Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("✅ Translation Glasses started")
        print("   Press 'q' to quit")
        
        # Main loop
        self._main_loop()
        
    def _run_with_test_image(self):
        """
        Run the application with a test image when camera is not available
        """
        print("Running with test image due to camera unavailability")
        
        # Load test image
        test_image_path = os.path.join(os.path.dirname(__file__), "test_images", "test_image.jpg")
        if not os.path.exists(test_image_path):
            print(f"❌ Test image not found at {test_image_path}")
            return
            
        frame = cv2.imread(test_image_path)
        if frame is None:
            print(f"❌ Failed to load test image from {test_image_path}")
            return
            
        print(f"✅ Loaded test image from {test_image_path}")
        
        # Process the image
        processed_frame = self.image_processor.process_frame(frame)
        
        # Perform OCR
        print("Performing OCR...")
        detections = self.ocr_engine.detect_text(processed_frame)
        print(f"Found {len(detections)} text regions")
        
        # Translate detected text
        translations = {}
        for _, text, _ in detections:
            if text.strip():
                translated = self.translator.translate_text(text)
                translations[text] = translated
                print(f"'{text}' -> '{translated}'")
        
        # Draw UI elements
        result_frame = self.ui.draw_text_detection(processed_frame, detections, translations)
        result_frame = self.ui.draw_controls(result_frame)
        
        # Save the result
        output_path = os.path.join(os.path.dirname(__file__), "test_images", "result_live.jpg")
        cv2.imwrite(output_path, result_frame)
        print(f"✅ Result saved to {output_path}")
        
        # Try to display the result if possible
        try:
            cv2.imshow("Wearable Translation Glasses (Test Mode)", result_frame)
            print("Press any key to exit")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Note: Could not display result window: {e}")
            print("Result has been saved to file instead.")
        
    def _main_loop(self):
        """
        Main application loop
        """
        while self.running:
            # Read frame from camera
            ret, frame = self.camera.read_frame()
            if not ret:
                print("❌ Failed to read frame")
                time.sleep(0.1)
                continue
                
            # Add frame to processing queue if not full
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
                
            # Get processed results if available
            if not self.result_queue.empty():
                result_frame, detections, translations, stats = self.result_queue.get()
                
                # Update statistics
                self.stats.update(stats)
                
                # Draw UI elements
                result_frame = self.ui.draw_text_detection(result_frame, detections, translations)
                result_frame = self.ui.draw_performance_stats(result_frame, self.stats)
                
                if not self.show_settings:
                    result_frame = self.ui.draw_controls(result_frame)
                else:
                    # Show settings view
                    settings_options = {
                        'Camera': [str(cam) for cam in self.camera.get_available_cameras()],
                        'Source Language': list(self.translator.get_supported_languages().keys()),
                        'Target Language': list(self.translator.get_supported_languages().keys())
                    }
                    result_frame = self.ui.create_settings_view(
                        result_frame, self.current_settings, settings_options
                    )
                
                # Display the result
                cv2.imshow("Wearable Translation Glasses", result_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('d'):
                self.ui.show_debug = not self.ui.show_debug
            elif key == ord('f'):
                self.ui.show_fps = not self.ui.show_fps
            elif key == ord('c'):
                self._cycle_camera()
            elif key == ord('s'):
                self._toggle_setting('Source Language')
            elif key == ord('t'):
                self._toggle_setting('Target Language')
            elif key == ord('r'):
                self.image_processor = ImageProcessor(stabilization_strength=0.8)
            elif key == 27:  # ESC
                self.show_settings = False
                
        # Cleanup
        self.camera.release()
        cv2.destroyAllWindows()
        print("✅ Application stopped")
        
    def _process_frames(self):
        """
        Process frames in a separate thread
        """
        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue
                
            # Get frame from queue
            frame = self.frame_queue.get()
            
            # Process image
            start_time = time.time()
            processed_frame = self.image_processor.process_frame(frame)
            processing_time = time.time() - start_time
            
            # Perform OCR
            start_time = time.time()
            detections = self.ocr_engine.detect_text(processed_frame)
            ocr_time = time.time() - start_time
            
            # Translate detected text
            translations = {}
            start_time = time.time()
            for _, text, _ in detections:
                if text.strip():
                    translations[text] = self.translator.translate_text(text)
            translation_time = time.time() - start_time
            
            # Update statistics
            stats = {
                'ocr_time': ocr_time,
                'translation_time': translation_time,
                'processing_time': processing_time,
                'detections': len(detections)
            }
            
            # Add results to output queue if not full
            if not self.result_queue.full():
                self.result_queue.put((processed_frame, detections, translations, stats))
                
            self.frame_queue.task_done()
            
    def _cycle_camera(self):
        """
        Cycle to the next available camera
        """
        available_cameras = self.camera.get_available_cameras()
        if not available_cameras:
            print("❌ No cameras available")
            return
            
        # Find next camera in the list
        current_idx = available_cameras.index(self.current_settings['Camera']) \
            if self.current_settings['Camera'] in available_cameras else -1
        next_idx = (current_idx + 1) % len(available_cameras)
        next_camera = available_cameras[next_idx]
        
        # Release current camera and start new one
        self.camera.release()
        if self.camera.start(next_camera):
            self.current_settings['Camera'] = next_camera
            print(f"✅ Switched to camera {next_camera}")
        else:
            print(f"❌ Failed to switch to camera {next_camera}")
            # Try to restart the previous camera
            self.camera.start(self.current_settings['Camera'])
            
    def _toggle_setting(self, setting_name):
        """
        Toggle a setting value
        
        Args:
            setting_name: Name of the setting to toggle
        """
        if setting_name == 'Source Language':
            languages = list(self.translator.get_supported_languages().keys())
            current_idx = languages.index(self.current_settings[setting_name]) \
                if self.current_settings[setting_name] in languages else 0
            next_idx = (current_idx + 1) % len(languages)
            self.current_settings[setting_name] = languages[next_idx]
            self.translator.set_languages(
                self.current_settings['Source Language'],
                self.current_settings['Target Language']
            )
            print(f"✅ Source language set to {languages[next_idx]}")
            
        elif setting_name == 'Target Language':
            languages = list(self.translator.get_supported_languages().keys())
            # Remove 'auto' from target language options
            if 'auto' in languages:
                languages.remove('auto')
                
            current_idx = languages.index(self.current_settings[setting_name]) \
                if self.current_settings[setting_name] in languages else 0
            next_idx = (current_idx + 1) % len(languages)
            self.current_settings[setting_name] = languages[next_idx]
            self.translator.set_languages(
                self.current_settings['Source Language'],
                self.current_settings['Target Language']
            )
            print(f"✅ Target language set to {languages[next_idx]}")
            
        elif setting_name == 'Stabilization':
            if self.current_settings[setting_name] == 'On':
                self.current_settings[setting_name] = 'Off'
                self.image_processor = ImageProcessor(stabilization_strength=0.0)
            else:
                self.current_settings[setting_name] = 'On'
                self.image_processor = ImageProcessor(stabilization_strength=0.8)
            print(f"✅ Stabilization {self.current_settings[setting_name]}")

def main():
    """
    Main entry point
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Wearable Translation Glasses')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--source', type=str, default='auto', help='Source language code (default: auto)')
    parser.add_argument('--target', type=str, default='en', help='Target language code (default: en)')
    parser.add_argument('--width', type=int, default=1280, help='Display width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Display height (default: 720)')
    parser.add_argument('--test', action='store_true', help='Run in test mode with a static image')
    args = parser.parse_args()
    
    if args.test:
        run_test_mode(args.source, args.target)
    else:
        # Create and start the application
        app = TranslationGlasses(
            camera_id=args.camera,
            source_lang=args.source,
            target_lang=args.target,
            width=args.width,
            height=args.height
        )
        app.start()

def run_test_mode(source_lang='auto', target_lang='en'):
    """
    Run the application in test mode with a static image
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
    """
    print("Running in test mode with a static image")
    print(f"Source language: {source_lang}")
    print(f"Target language: {target_lang}")
    
    # Load test image
    test_image_path = os.path.join(os.path.dirname(__file__), "test_images", "test_image.jpg")
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found at {test_image_path}")
        return
        
    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"❌ Failed to load test image from {test_image_path}")
        return
        
    print(f"✅ Loaded test image from {test_image_path}")
    print(f"Image dimensions: {frame.shape[1]}x{frame.shape[0]}")
    
    # Initialize components
    print("Initializing components...")
    image_processor = ImageProcessor(stabilization_strength=0.8)
    ocr_engine = OCREngine(languages=['en'], stabilization_frames=1)
    translator = Translator(source_lang, target_lang)
    ui = UIManager(show_fps=True, show_controls=True)
    print("✅ Components initialized")
    
    # Process the image
    print("Processing image...")
    processed_frame = image_processor.process_frame(frame)
    print("✅ Image processed")
    
    # Perform OCR
    print("Performing OCR...")
    detections = ocr_engine.detect_text(processed_frame)
    print(f"✅ OCR completed. Found {len(detections)} text regions:")
    for i, (bbox, text, conf) in enumerate(detections):
        print(f"  {i+1}. '{text}' (Confidence: {conf:.2f})")
    
    # Translate detected text
    print("\nTranslating text...")
    translations = {}
    for _, text, _ in detections:
        if text.strip():
            translated = translator.translate_text(text)
            translations[text] = translated
            print(f"  '{text}' -> '{translated}'")
    print("✅ Translation completed")
    
    # Draw UI elements
    print("\nDrawing UI elements...")
    result_frame = ui.draw_text_detection(processed_frame, detections, translations)
    result_frame = ui.draw_controls(result_frame)
    print("✅ UI elements drawn")
    
    # Save the result
    output_path = os.path.join(os.path.dirname(__file__), "test_images", "result.jpg")
    cv2.imwrite(output_path, result_frame)
    print(f"✅ Result saved to {output_path}")
    
    # Try to display the result if possible
    try:
        cv2.imshow("Wearable Translation Glasses (Test Mode)", result_frame)
        print("Press any key to exit")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Note: Could not display result window: {e}")
        print("Result has been saved to file instead.")

if __name__ == "__main__":
    main()
