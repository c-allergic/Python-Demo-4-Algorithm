import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term (x0 = 1)
X_b = np.c_[np.ones((100, 1)), X]

# Hyperparameters
eta = 0.07  # Learning rate
max_iterations = 200
m = 100

# Initialize parameters
w = np.random.randn(2, 1)

# Store cost history for visualization
cost_history = []
converged_at = 0

for iteration in range(max_iterations):
    # use 1/m as the parameter of cost function
    gradients = 2/m * X_b.T.dot(X_b.dot(w) - y)
    w = w - eta * gradients
    cost = 1/(2*m) * np.sum((X_b.dot(w) - y)**2)
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Cost = {cost}")
        print(f"Parameters: {w.T}")
    if iteration>0 and abs(cost-cost_history[-1]) < 1e-4:
        print(f"Converged at iteration {iteration}")
        converged_at = iteration
        break
    cost_history.append(cost)
    
converged_at = max_iterations if converged_at == 0 else converged_at
# Plot cost function over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(converged_at), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Decreasing Over Iterations")
plt.show()

# Final parameters
w_best = w
print(f"Optimal parameters: {w_best.T}")
