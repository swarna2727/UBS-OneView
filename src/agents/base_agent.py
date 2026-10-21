"""Simplified Base Agent for Investment Strategy Pipeline."""

import time
from typing import Dict, Any
from abc import ABC, abstractmethod

from src.utils.logger import setup_logger
from src.utils.gemini_client import GeminiClient

class BaseAgent(ABC):
    """Simplified base class for all pipeline agents."""
    
    def __init__(self, name: str, api_key: str):
        self.name = name
        self.api_key = api_key
        self.gemini_client = GeminiClient(api_key)
        self.logger = setup_logger(name)
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the data and return enriched data.
        
        Args:
            data: The master JSON schema data
            
        Returns:
            The data with this agent's output added
        """
        pass
    
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point - adds timing and error handling.
        
        Args:
            data: The master JSON schema data
            
        Returns:
            The data with this agent's output added
        """
        start_time = time.time()
        self.logger.info(f"Starting {self.name} processing")
        
        try:
            result = self.process(data)
            processing_time = round((time.time() - start_time) * 1000, 2)
            
            # Add processing time to the result
            if isinstance(result, dict):
                result['processing_time_ms'] = processing_time
            
            self.logger.info(f"{self.name} completed successfully in {processing_time}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"{self.name} failed: {str(e)}")
            raise