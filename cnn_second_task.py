import numpy as np
def tanh(x):
    return np.tanh(x)
def tanh_dev(x):
    return 1 - np.tanh(x) ** 2
inputs = np.array([[0.05, 0.10]])
y_expected = np.array([[0.01, 0.99]])
weights_input_hidden = np.array([[0.15, 0.25], [0.20, 0.30]])
weights_hidden_output = np.array([[0.40, 0.50], [0.45, 0.55]])
bias_hidden = np.array([0.35, 0.35])
bias_output = np.array([0.60, 0.60])
learning_rate = 0.5
epochs = 10000
for epoch in range(epochs):
    hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_output = tanh(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = tanh(final_input)
    error = y_expected - final_output
    output_gradient = error * tanh_dev(final_output)
    hidden_error = output_gradient.dot(weights_hidden_output.T)
    hidden_gradient = hidden_error * tanh_dev(hidden_output)
    weights_hidden_output += np.dot(hidden_output.T, output_gradient) * learning_rate
    weights_input_hidden += np.dot(inputs.T, hidden_gradient) * learning_rate
    bias_output += np.sum(output_gradient, axis=0) * learning_rate
    bias_hidden += np.sum(hidden_gradient, axis=0) * learning_rate
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch}, Loss: {loss}")
print("Final Output:")
print(final_output)
