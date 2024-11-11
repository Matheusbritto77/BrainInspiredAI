# CoreModules/Memory/ShortTermMemory.py
import torch
from collections import deque
import numpy as np
from datetime import datetime
from Utils.Logger import setup_logger

class ShortTermMemoryCache:
    def __init__(self, max_size: int = 1000):
        self.logger = setup_logger()
        self.memory_buffer = deque(maxlen=max_size)
        self.attention_weights = torch.nn.Parameter(torch.ones(max_size))
        self.temporal_decay = 0.95
        
    def store(self, input_data: str, emotional_context: dict):
        """
        Armazena nova informação na memória de curto prazo
        """
        try:
            memory_item = {
                'content': input_data,
                'emotional_context': emotional_context,
                'timestamp': datetime.now(),
                'attention_score': 1.0
            }
            
            self.memory_buffer.append(memory_item)
            self._update_attention_weights()
            
        except Exception as e:
            self.logger.error(f"Error storing in short-term memory: {str(e)}")
    
    def _update_attention_weights(self):
        """
        Atualiza os pesos de atenção baseado na relevância temporal
        """
        try:
            current_time = datetime.now()
            for i, item in enumerate(self.memory_buffer):
                time_diff = (current_time - item['timestamp']).total_seconds()
                decay_factor = self.temporal_decay ** (time_diff / 3600)  # Decay por hora
                self.attention_weights[i] *= decay_factor
                
        except Exception as e:
            self.logger.error(f"Error updating attention weights: {str(e)}")

# CoreModules/Memory/LongTermMemory.py
import sqlite3
import torch
import pickle
from datetime import datetime
from Utils.Logger import setup_logger

class LongTermMemory:
    def __init__(self, db_path: str = 'Database/LongTermMemoryStorage.db'):
        self.logger = setup_logger()
        self.db_path = db_path
        self.initialize_db()
        
    def initialize_db(self):
        """
        Inicializa o banco de dados de memória de longo prazo
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Criação da tabela de memórias
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    emotional_context BLOB,
                    embedding BLOB,
                    importance_score REAL,
                    access_count INTEGER DEFAULT 0,
                    last_access TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error initializing long-term memory DB: {str(e)}")
    
    async def store(self, content: str, emotional_context: dict, embedding: torch.Tensor):
        """
        Armazena uma nova memória no banco de dados
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialização dos dados
            emotional_context_blob = pickle.dumps(emotional_context)
            embedding_blob = pickle.dumps(embedding.detach().numpy())
            
            cursor.execute('''
                INSERT INTO memories (content, emotional_context, embedding, 
                                    importance_score, last_access)
                VALUES (?, ?, ?, ?, ?)
            ''', (content, emotional_context_blob, embedding_blob, 
                  self._calculate_importance(emotional_context), datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing in long-term memory: {str(e)}")
    
    async def retrieve(self, query_embedding: torch.Tensor, k: int = 5):
        """
        Recupera as k memórias mais similares
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Recuperação de todas as memórias
            cursor.execute('SELECT id, embedding, content, emotional_context FROM memories')
            memories = cursor.fetchall()
            
            # Cálculo de similaridade
            similarities = []
            for mem_id, emb_blob, content, emotional_context_blob in memories:
                stored_embedding = torch.tensor(pickle.loads(emb_blob))
                similarity = torch.cosine_similarity(query_embedding, stored_embedding, dim=0)
                similarities.append((similarity.item(), mem_id, content, emotional_context_blob))
            
            # Ordenação por similaridade
            similarities.sort(reverse=True)
            top_memories = similarities[:k]
            
            # Atualização dos contadores de acesso
            for _, mem_id, _, _ in top_memories:
                cursor.execute('''
                    UPDATE memories 
                    SET access_count = access_count + 1,
                        last_access = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (mem_id,))
            
            conn.commit()
            conn.close()
            
            return [(content, pickle.loads(ec_blob)) 
                    for _, _, content, ec_blob in top_memories]
            
        except Exception as e:
            self.logger.error(f"Error retrieving from long-term memory: {str(e)}")
            return []
    
    def _calculate_importance(self, emotional_context: dict) -> float:
        """
        Calcula a importância da memória baseada no contexto emocional
        """
        try:
            # Implementação do cálculo de importância
            importance = abs(emotional_context.get('polarity', 0)) + \
                        emotional_context.get('confidence', 0)
            return min(1.0, importance)
            
        except Exception as e:
            self.logger.error(f"Error calculating memory importance: {str(e)}")
            return 0.5