# main.py
import os
import torch
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from CoreModules.CortexPreFrontal.DecisionMaking import DecisionEngine
from CoreModules.Hippocampus.MemoryFormation import MemoryFormatter
from CoreModules.Amygdala.EmotionalValues import EmotionAnalyzer
from CoreModules.Memory.ShortTermMemory import ShortTermMemoryCache
from Utils.Logger import setup_logger
from Utils.Config import load_config

class BrainInspiredAI:
    def __init__(self):
        self.config = load_config()
        self.logger = setup_logger()
        
        # Inicialização do modelo GPT-2 em português
        self.tokenizer = GPT2Tokenizer.from_pretrained('pierreguillou/gpt2-small-portuguese')
        self.model = GPT2LMHeadModel.from_pretrained('pierreguillou/gpt2-small-portuguese')
        
        # Inicialização dos módulos principais
        self.decision_engine = DecisionEngine()
        self.memory_formatter = MemoryFormatter()
        self.emotion_analyzer = EmotionAnalyzer()
        self.short_term_memory = ShortTermMemoryCache()
        
        self.logger.info("BrainInspiredAI initialized successfully")

    async def process_input(self, user_input: str) -> str:
        """
        Processa a entrada do usuário através de todos os módulos cerebrais
        """
        try:
            # Análise emocional do input
            emotional_context = await self.emotion_analyzer.analyze(user_input)
            
            # Formação de memória de curto prazo
            self.short_term_memory.store(user_input, emotional_context)
            
            # Recuperação de contexto relevante
            context = self.memory_formatter.retrieve_relevant_context(user_input)
            
            # Geração de resposta usando GPT-2
            inputs = self.tokenizer.encode(user_input + context, return_tensors='pt')
            outputs = self.model.generate(
                inputs,
                max_length=150,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7
            )
            
            # Processamento da resposta através do motor de decisão
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            final_response = await self.decision_engine.process_response(
                response, emotional_context
            )
            
            self.logger.info(f"Processed input successfully: {user_input[:50]}...")
            return final_response
            
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            return "Desculpe, ocorreu um erro ao processar sua mensagem."

    async def train(self, training_data):
        """
        Treina o modelo com novos dados
        """
        try:
            # Implementação do treinamento
            pass
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")

if __name__ == "__main__":
    brain = BrainInspiredAI()
    # Exemplo de uso
    import asyncio
    response = asyncio.run(brain.process_input("Olá, como você está?"))
    print(response)