import numpy as np

# Load a single array from .npy file
data = np.load('/home/skr/Downloads/055-AMP-Adversarial-Motion-Priors-for-Stylized-Physics-Based-Character-Control/assets/amp/motions/amp_humanoid_backflip.npy', allow_pickle=True)
print(data)