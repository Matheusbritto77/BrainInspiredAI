# Lógica de tomada de decisão e planejamento
import random

def make_decision():
    actions = ['Ação1', 'Ação2', 'Ação3']
    decision = random.choice(actions)
    print(f"Decisão tomada: {decision}")
    return decision

if __name__ == '__main__':
    make_decision()
