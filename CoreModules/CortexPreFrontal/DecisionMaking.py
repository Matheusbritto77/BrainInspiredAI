# L�gica de tomada de decis�o e planejamento
import random

def make_decision():
    actions = ['A��o1', 'A��o2', 'A��o3']
    decision = random.choice(actions)
    print(f"Decis�o tomada: {decision}")
    return decision

if __name__ == '__main__':
    make_decision()
