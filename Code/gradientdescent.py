import numpy as np
import matplotlib.pyplot as plt
import torch
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

# Final parameters
w_best = w
print(f"Optimal parameters: {w_best.T}")


# use torch interface to implement the same task
# Convert numpy arrays to PyTorch tensors
X_tensor = torch.from_numpy(X).float()  # Feature data
y_tensor = torch.from_numpy(y).float()  # Target values

# Add bias term (x0 = 1)
X_b_tensor = torch.cat([torch.ones(100, 1), X_tensor], dim=1)

# Hyperparameters
eta = 0.06  # Learning rate
max_iterations = 200  # Maximum number of iterations
m = 100  # Number of samples

# Initialize parameters with requires_grad=True to track gradients
w = torch.randn(2, 1, requires_grad=True)
print(f"Starting point: {w.T.tolist()[0]}")
# Store cost history for visualization
cost_history = []
converged_at = 0

for iteration in range(max_iterations):
    # Forward propagation: compute predictions
    predictions = X_b_tensor.mm(w)  # Matrix multiplication
    
    # Compute loss (Mean Squared Error)
    loss = (1/(m)) * torch.sum((predictions - y_tensor)**2)
    
    # Backward propagation: compute gradients
    loss.backward()  # Automatically compute gradients for tensors with requires_grad=True
    
    # Update parameters
    with torch.no_grad():  # Disable gradient tracking during parameter update
        w -= eta * w.grad
    
    # Zero the gradients after update
    w.grad.zero_()
    
    # Record the loss
    cost_history.append(loss.item())
    
    """
    # Print iteration information
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: Cost = {loss.item()}")
        print(f"Parameters: {w.T}")
    """
    
    # Check convergence condition
    if iteration > 0 and abs(loss.item() - cost_history[-2]) < 1e-4:
        converged_at = iteration+1
        break

# Set converged_at to max_iterations if not converged early
converged_at = max_iterations if converged_at == 0 else converged_at

# Final parameters
w_best = w
print(f"Optimal parameters: {w_best.T.tolist()[0]}")

# plotting
plt.figure(figsize=(10, 6))
plt.plot(range(converged_at), cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Function Decreasing Over Iterations")
plt.show()
