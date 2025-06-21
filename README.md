# Wearable Translation Glasses with AR Integration

Real-time OCR and translation system with stabilization to simulate AR glasses experience.

## Abstract

Wearable Translation Glasses with AR Integration offer real-time language translation by displaying translated text directly on the lenses using augmented reality. By combining voice recognition, OCR, and AR display technology, these smart glasses help users understand spoken and written foreign languages instantly. Ideal for travelers and language learners, they enhance communication and bridge language gaps in everyday situations.

The development of wearable translation glasses with AR integration marks a significant leap forward in real-time communication and accessibility technology. By combining advanced language translation algorithms with augmented reality, these glasses offer users an intuitive, hands-free solution for overcoming language barriers in various settings, from travel and education to business and healthcare. The seamless integration of real-time subtitles into the user's visual field enhances both comprehension and engagement, while maintaining a discreet and user-friendly experience. As the technology continues to evolve, future iterations hold the potential for even broader language support, improved contextual accuracy, and more compact, stylish designs—paving the way for a more connected, inclusive, and multilingual world.

## Features

- **Stable OCR**: Google Translate-like OCR stability that's not affected by minor head movements
- **Multi-camera Support**: Works with both built-in and external USB cameras
- **Real-time Translation**: Translates detected text in real-time
- **Image Stabilization**: Reduces the impact of camera movement on OCR accuracy
- **Modular Architecture**: Well-organized code structure for easy maintenance and extension
- **User-friendly Interface**: On-screen controls and settings

## Requirements

- Python 3.7+
- OpenCV
- EasyOCR
- NumPy
- Torch (for EasyOCR)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/wearable-translation-glasses.git
   cd wearable-translation-glasses
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main application:

```
python translation_glasses.py
```

### Command-line Options

- `--camera`: Camera index (default: 0)
- `--source`: Source language code (default: auto)
- `--target`: Target language code (default: en)
- `--width`: Display width (default: 1280)
- `--height`: Display height (default: 720)

Example:
```
python translation_glasses.py --camera 1 --target es
```

### Keyboard Controls

- `Q`: Quit the application
- `D`: Toggle debug information
- `F`: Toggle FPS display
- `C`: Cycle through available cameras
- `S`: Change source language
- `T`: Change target language
- `R`: Reset image stabilization

## Project Structure

- `translation_glasses.py`: Main application file
- `modules/`: Directory containing modular components
  - `camera_manager.py`: Camera handling and selection
  - `image_processor.py`: Image enhancement and stabilization
  - `ocr_engine.py`: Text detection with stabilization
  - `translator.py`: Text translation between languages
  - `ui_manager.py`: User interface elements

## How It Works

1. The camera captures video frames
2. Frames are processed for stabilization and enhancement
3. OCR detects text in the processed frames
4. Detected text is translated to the target language
5. Results are displayed with the original text and translation

The system uses a multi-threaded approach to maintain smooth performance, with separate threads for frame capture, processing, and display.
