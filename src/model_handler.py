from typing import Dict, Tuple, Union

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

class ToxicityPredictor:
    """Predicts toxicity in text using ONNX model from Hugging Face."""

    def __init__(self, cache_size: int = 1000) -> None:
        """Initialize toxicity prediction model from Hugging Face.

        Args:
            cache_size: Number of recent predictions to cache.

        Raises:
            RuntimeError: If model download or initialization fails.
        """
        try:
            model_path = hf_hub_download(
                repo_id='karthikarunr/BullyGuard',
                filename='cyberbullying_model.onnx'
            )
            
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.cache = {}
            self.cache_size = cache_size
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")
        
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
