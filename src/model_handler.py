import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
from typing import Tuple, Dict, Union
import os
# from pathlib import Path

class ToxicityPredictor:
    def __init__(self, model_path: str, cache_size: int = 1000):
        """
        Initialize the toxicity prediction model with ONNX runtime.
        
        Args:
            model_path: Path to the ONNX model file
            cache_size: Number of recent predictions to cache
        
        Usage:
            predictor = ToxicityPredictor('models/cyberbullying_model.onnx')
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.cache = {}
        self.cache_size = cache_size
        
    def predict(self, text: str) -> Tuple[bool, float]:
        """
        Predict toxicity of input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple containing:
            - Boolean indicating if text is toxic (True) or not (False)
            - Float confidence score between 0 and 1
            
        Usage:
            is_toxic, confidence = predictor.predict("some text")
        """
        if text in self.cache:
            return self.cache[text]
            
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        ort_inputs = {
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy()
        }
        
        try:
            ort_outputs = self.session.run(None, ort_inputs)
            probability = 1 / (1 + np.exp(-ort_outputs[0]))  # sigmoid
            is_toxic = bool(probability > 0.5)
            confidence = float(probability)
            
            # Update cache
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[text] = (is_toxic, confidence)
            
            return is_toxic, confidence
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def batch_predict(self, texts: list) -> list[Tuple[bool, float]]:
        """
        Predict toxicity for multiple texts.
        
        Args:
            texts: List of strings to analyze
            
        Returns:
            List of tuples, each containing:
            - Boolean indicating if text is toxic (True) or not (False)
            - Float confidence score between 0 and 1
            
        Usage:
            results = predictor.batch_predict(["text1", "text2"])
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        return results
    
    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get statistics about the prediction cache.
        
        Returns:
            Dictionary containing:
            - cache_size: Current number of cached predictions
            - cache_limit: Maximum cache size
            - cache_usage: Percentage of cache used
            
        Usage:
            stats = predictor.get_cache_stats()
        """
        return {
            'cache_size': len(self.cache),
            'cache_limit': self.cache_size,
            'cache_usage': len(self.cache) / self.cache_size * 100
        }
