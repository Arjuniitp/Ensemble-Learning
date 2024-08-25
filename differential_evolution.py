import numpy as np
from scipy.optimize import differential_evolution

# Define the evaluation function for differential evolution
def evaluate_de(params):
    top, alpha_1, alpha_2, alpha_3 = map(int, np.round(params))
    predictions = Gompertz(top, alpha_1, alpha_2, alpha_3, p1, p2, p3, p4, p5)
    predictions = np.squeeze(predictions)
    correct = np.sum(predictions == labels)
    total = labels.shape[0]
    accuracy = correct / total
    return -accuracy  # Minimize negative accuracy

# Define the bounds for each parameter
bounds = [(1, 5), (1, 10), (1, 10), (1, 10)]

# Run the differential evolution algorithm
result = differential_evolution(evaluate_de, bounds, strategy='best1bin', maxiter=100, popsize=100, tol=1e-6)

# Print the result
best_params = result.x
best_accuracy = -result.fun
print(f"Best parameters: {best_params}")
