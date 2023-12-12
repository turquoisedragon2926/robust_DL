import torch
import random
from torch import linalg
import torch.nn.functional as F

class NoiseFunction:
    def __init__(self, severity):
        self.severity = severity
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add_noise(self, data):
        raise NotImplementedError

class GaussianNoise(NoiseFunction):
    def add_noise(self, data):
        noise = torch.randn(data.size()) * self.severity
        noise = torch.Tensor(noise).to(self.device)

        noisy_data = data+noise
        noise_norm = linalg.vector_norm(noise, dim = (1,2,3)).reshape((-1,1)) # improve this line of code later
        return noisy_data, noise_norm

class UniformNoise(NoiseFunction):
    def add_noise(self, data):
        # Uniform noise in the range [-severity, severity]
        noise = (torch.rand(data.size()) * 2 - 1) * self.severity
        noise = noise.to(self.device)

        noisy_data = data + noise
        noise_norm = linalg.vector_norm(noise, dim=(1,2,3)).reshape((-1,1))
        return noisy_data, noise_norm

class ShotNoise(NoiseFunction):
    def add_noise(self, data):
        # The severity factor can be used to scale the noise
        max_val = data.max()
        noise = torch.poisson(data * self.severity) / self.severity - data
        noise *= max_val / noise.max()  # Optional: scale the noise to match the max value of the data

        noise = noise.to(self.device)

        noisy_data = data + noise
        noise_norm = linalg.vector_norm(noise, dim=(1,2,3)).reshape((-1,1))
        return noisy_data, noise_norm

class BlurNoise(NoiseFunction):
    def add_noise(self, data):
        # Define a 5x5 Gaussian blur kernel
        kernel_size = 5
        sigma = self.severity

        # Create a 1D Gaussian kernel
        kernel_1d = torch.tensor([1.0], device=self.device).new_full((kernel_size,), 1.0)
        for i in range(1, (kernel_size // 2) + 1):
            value = torch.exp(-0.5 * (i / sigma)**2)
            kernel_1d[kernel_size // 2 - i] = value
            kernel_1d[kernel_size // 2 + i] = value

        # Normalize the kernel so that it sums to one
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Create a 2D Gaussian kernel from the 1D kernel
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]

        # Add channel dimension
        kernel_2d = kernel_2d.expand(data.size(1), 1, kernel_size, kernel_size)

        # Apply padding to maintain the size
        padding = kernel_size // 2

        # Apply the Gaussian blur filter
        blurred_data = F.conv2d(data, kernel_2d, padding=padding, groups=data.size(1))
        noise = blurred_data - data
        noise_norm = linalg.vector_norm(noise, dim=(1,2,3)).reshape((-1,1))

        return blurred_data, noise_norm

class NoiseFunctionFactory:
    @staticmethod
    def get_noise_function(train_noise, severity):
        noise_classes = {
            'gaussian': GaussianNoise,
            'uniform': UniformNoise,
            'shot': ShotNoise,
            'blur': BlurNoise
        }
        if train_noise == 'random':
            return random.choice(list(noise_classes.values()))(severity)
        elif train_noise in noise_classes:
            return noise_classes[train_noise](severity)
        else:
            raise ValueError("Noise type not implemented")
