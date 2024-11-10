# Aprendizado associativo com redes neurais simples
import numpy as np

def associative_learning(inputs, target):
    weights = np.random.rand(len(inputs))
    output = np.dot(inputs, weights)
    error = target - output
    weights += 0.1 * error * np.array(inputs)
    print(f'Pesos ajustados: {weights}')
    return weights

if __name__ == '__main__':
    inputs = [0.5, 0.2, 0.1]
    target = 0.8
    associative_learning(inputs, target)
