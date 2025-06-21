"""
Translator Module for Wearable Translation Glasses
Handles text translation between languages
"""

import time
from typing import Dict, List, Optional
import requests
import json
import os

class Translator:
    def __init__(self, 
                 source_lang: str = 'auto',
                 target_lang: str = 'en',
                 cache_size: int = 100):
        """
        Initialize the translator
        
        Args:
            source_lang: Source language code or 'auto' for auto-detection
            target_lang: Target language code
            cache_size: Maximum number of translations to cache
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.cache_size = cache_size
        self.translation_cache = {}
        self.supported_languages = self._get_supported_languages()
        
    def _get_supported_languages(self) -> Dict[str, str]:
        """
        Get dictionary of supported languages
        
        Returns:
            Dictionary mapping language codes to language names
        """
        # Common language codes
        return {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'auto': 'Auto-detect'
        }
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get dictionary of supported languages
        
        Returns:
            Dictionary mapping language codes to language names
        """
        return self.supported_languages
    
    def set_languages(self, source_lang: str, target_lang: str) -> None:
        """
        Set source and target languages
        
        Args:
            source_lang: Source language code or 'auto'
            target_lang: Target language code
        """
        if source_lang in self.supported_languages:
            self.source_lang = source_lang
        
        if target_lang in self.supported_languages:
            self.target_lang = target_lang
    
    def translate_text(self, text: str) -> str:
        """
        Translate text from source language to target language
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        # Skip empty text
        if not text or text.strip() == '':
            return ''
            
        # Check cache first
        cache_key = f"{self.source_lang}:{self.target_lang}:{text}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
            
        try:
            # Use LibreTranslate API (you can replace with other translation APIs)
            # This is a placeholder - in production you would use a proper API
            translated = self._mock_translation(text)
            
            # Cache the result
            if len(self.translation_cache) >= self.cache_size:
                # Remove oldest entry if cache is full
                self.translation_cache.pop(next(iter(self.translation_cache)))
                
            self.translation_cache[cache_key] = translated
            return translated
            
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text on error
    
    def _mock_translation(self, text: str) -> str:
        """
        Mock translation function for testing
        In production, replace with actual API call
        
        Args:
            text: Text to translate
            
        Returns:
            Translated text
        """
        # This is just a placeholder for testing
        # In a real implementation, you would call an actual translation API
        
        # Simulate translation delay
        time.sleep(0.1)
        
        # Simple mock translations for demo purposes
        if self.target_lang == 'es':
            translations = {
                'hello': 'hola',
                'world': 'mundo',
                'welcome': 'bienvenido',
                'thank you': 'gracias',
                'goodbye': 'adiós'
            }
            
            for eng, span in translations.items():
                if eng.lower() in text.lower():
                    return text.lower().replace(eng, span)
        
        # For other languages or unknown text, just return the original
        # with a language indicator
        return f"[{self.target_lang}] {text}"
