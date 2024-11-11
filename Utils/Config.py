# Utils/Config.py
import yaml
import os
from pathlib import Path
from typing import Dict, Any

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """
        Inicializa o gerenciador de configuração
        """
        self.config = {
            "model": {
                "name": "gpt2-small-portuguese",
                "max_length": 150,
                "temperature": 0.7
            },
            "memory": {
                "short_term_size": 1000,
                "long_term_db_path": "Database/LongTermMemoryStorage.db"
            },
            "learning": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 10
            },
            "neural_oscillations": {
                "base_frequency": 100,
                "alpha_range": (8, 13),
                "beta_range": (13, 30),
                "gamma_range": (30, 100)
            },
            "sleep": {
                "cycle_duration": 3600,
                "stage_duration": 900
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Recupera um valor de configuração
        """
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default

def load_config() -> ConfigManager:
    """
    Carrega e retorna o gerenciador de configuração
    """
    return ConfigManager()

# Utils/Logger.py
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logger(name: str = "BrainInspiredAI") -> logging.Logger:
    """
    Configura e retorna um logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Utils/DataPreprocessing.py
import torch
import numpy as np
from typing import Union, List
from transformers import PreTrainedTokenizer

class DataPreprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.logger = setup_logger()
        self.tokenizer = tokenizer
        
    def preprocess_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Pré-processa texto para uso no modelo
        """
        try:
            # Tokenização
            tokens = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            return tokens
            
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {str(e)}")
            return None
    
    def clean_text(self, text: str) -> str:
        """
        Limpa e normaliza texto
        """
        try:
            # Implementação da limpeza de texto
            cleaned_text = text.strip().lower()
            return cleaned_text
            
        except Exception as e:
            self.logger.error(f"Error cleaning text: {str(e)}")
            return text