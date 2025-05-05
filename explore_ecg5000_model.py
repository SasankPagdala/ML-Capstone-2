import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = keras.models.load_model('ecg5000_model_final.keras')

# Print model summary
print("Model architecture:")
model.summary()

# Get model configuration
print("\nModel config:")
config = model.get_config()
print(config)

# List the weights
print("\nModel weights:")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}, Weights shapes: {[w.shape for w in layer.get_weights()]}")

# If you have the dataset, you can also visualize predictions
# For example, if you have test data:
# test_data = ...
# predictions = model.predict(test_data)
# print("Predictions sample:", predictions[:5])

# Visualize the model architecture (optional)
try:
    from tensorflow.keras.utils import plot_model
    plot_model(model, to_file='model_architecture.png', show_shapes=True)
    print("Model architecture visualization saved as model_architecture.png")
except ImportError:
    print("Could not import plot_model to visualize architecture")