# OscillationsAndRhythms/RhythmGenerator.py
import numpy as np
import torch
from Utils.Logger import setup_logger
from typing import Dict, Optional

class RhythmGenerator:
    def __init__(self):
        self.logger = setup_logger()
        self.active_rhythms = {}
        self.base_frequency = 100  # Hz
        
    async def generate_rhythm(self, rhythm_type: str, duration: float) -> Optional[torch.Tensor]:
        """
        Gera um padrão rítmico específico
        """
        try:
            time = torch.linspace(0, duration, int(duration * self.base_frequency))
            
            if rhythm_type == "alpha":
                return self._generate_alpha_rhythm(time)
            elif rhythm_type == "beta":
                return self._generate_beta_rhythm(time)
            elif rhythm_type == "gamma":
                return self._generate_gamma_rhythm(time)
            else:
                raise ValueError(f"Unsupported rhythm type: {rhythm_type}")
                
        except Exception as e:
            self.logger.error(f"Error generating rhythm: {str(e)}")
            return None
    
    def _generate_alpha_rhythm(self, time: torch.Tensor) -> torch.Tensor:
        """
        Gera ritmo alfa (8-13 Hz)
        """
        frequency = 10  # Hz
        return torch.sin(2 * np.pi * frequency * time)
    
    def _generate_beta_rhythm(self, time: torch.Tensor) -> torch.Tensor:
        """
        Gera ritmo beta (13-30 Hz)
        """
        frequency = 20  # Hz
        return torch.sin(2 * np.pi * frequency * time)
    
    def _generate_gamma_rhythm(self, time: torch.Tensor) -> torch.Tensor:
        """
        Gera ritmo gama (30-100 Hz)
        """
        frequency = 40  # Hz
        return torch.sin(2 * np.pi * frequency * time)

# OscillationsAndRhythms/BrainwaveModulation.py
class BrainwaveModulator:
    def __init__(self):
        self.logger = setup_logger()
        self.rhythm_generator = RhythmGenerator()
        self.current_state = "relaxed"
        
    async def modulate_state(self, target_state: str) -> bool:
        """
        Modula o estado do sistema através de padrões de onda cerebrais
        """
        try:
            # Define parâmetros de modulação
            modulation_params = self._get_modulation_params(target_state)
            
            # Gera os ritmos necessários
            rhythms = {}
            for rhythm_type, duration in modulation_params['rhythms'].items():
                rhythm = await self.rhythm_generator.generate_rhythm(
                    rhythm_type, duration
                )
                rhythms[rhythm_type] = rhythm
            
            # Aplica a modulação
            success = await self._apply_modulation(rhythms, modulation_params)
            
            if success:
                self.current_state = target_state
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error in brainwave modulation: {str(e)}")
            return False
    
    def _get_modulation_params(self, target_state: str) -> dict:
        """
        Define parâmetros de modulação para o estado alvo
        """
        params = {
            "focused": {
                "rhythms": {"beta": 2.0, "gamma": 1.0},
                "intensity": 0.8
            },
            "relaxed": {
                "rhythms": {"alpha": 3.0},
                "intensity": 0.6
            },
            "learning": {
                "rhythms": {"theta": 2.0, "gamma": 1.0},
                "intensity": 0.7
            }
        }
        return params.get(target_state, params["relaxed"])

# OscillationsAndRhythms/StateOptimizer.py
class StateOptimizer:
    def __init__(self):
        self.logger = setup_logger()
        self.modulator = BrainwaveModulator()
        self.optimal_states = {
            "learning": {"beta": 0.6, "gamma": 0.4},
            "processing": {"gamma": 0.8},
            "memory_consolidation": {"theta": 0.7, "alpha": 0.3}
        }
        
    async def optimize_for_task(self, task_type: str) -> bool:
        """
        Otimiza o estado do sistema para um tipo específico de tarefa
        """
        try:
            if task_type not in self.optimal_states:
                self.logger.warning(f"Unknown task type: {task_type}")
                return False
            
            # Obtém estado ótimo para a tarefa
            target_state = self.optimal_states[task_type]
            
            # Aplica modulação para atingir estado ótimo
            success = await self.modulator.modulate_state(task_type)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error optimizing state: {str(e)}")
            return False