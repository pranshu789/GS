import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift


# Defining an initial field E(x, y) in the near field
def gaussian(x, y, sigma):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

# Grid setup
N = 256  # Spatial resolution
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)

# Setting up a sample Gaussian field
sigma_nf = 0.5
A_nf = gaussian(X, Y, sigma_nf)
phi_nf = np.ones_like(A_nf)  # Unity phase
E_nf = A_nf * np.exp(1j * phi_nf)

# Testing Fourier Transform back and forth
E_ft = fft2(E_nf)
E_nf_back = ifft2(E_ft)

# Display the results
plt.figure(figsize=(12, 4))

# Near field
plt.subplot(131)
plt.imshow(np.abs(E_nf), cmap='Greens')
plt.title('Near Field (Original)')
plt.colorbar()

# Far field
plt.subplot(132)
plt.imshow(np.abs(E_ft), cmap='Greens')
plt.title('Far Field (Fourier Transform)')
plt.colorbar()

# Near field again
plt.subplot(133)
plt.imshow(np.abs(E_nf_back), cmap='Greens')
plt.title('Near Field (Back)')
plt.colorbar()

plt.tight_layout()
plt.show()

# Fourier transform back and forth works!!

# Introduce A_target to be a smiley pattern
A_target = np.zeros_like(E_ft)

# Circle contour
circle = ((X**2 + Y**2) < 0.20) & ((X**2 + Y**2) > 0.15)
A_target[circle] = 1

circle2 = ((X**2 + Y**2) < 0.05)
A_target[circle2] = 1

# Two eyes
eye1 = ((X + 0.3) ** 2 + (Y + 0.3) ** 2) < 0.05
eye2 = ((X - 0.3) ** 2 + (Y + 0.3) ** 2) < 0.05
A_target[eye1] = 1
A_target[eye2] = 1

# Normalize A_target to avoid scaling issues
A_target = A_target / np.max(A_target)

# Update the far field with the target amplitude
E_ft_updated = A_target * np.exp(1j * np.angle(E_ft))

# Perform inverse Fourier transform
E_nf_updated = ifft2(E_ft_updated)

E_nf_updated = A_nf * np.exp(1j * np.angle(E_nf_updated))

# Plotting
plt.figure(figsize=(15, 5))

# Original smiley
plt.subplot(1, 4, 1)
plt.imshow(np.abs(A_target), cmap='Greens')
plt.title('Original Smiley')

# First Gaussian
plt.subplot(1, 4, 2)
plt.imshow(np.abs(A_nf), cmap='Greens')
plt.title('Original Gaussian in NF')

# Target amplitude in FT
plt.subplot(1, 4, 3)
plt.imshow(np.abs(E_ft_updated), cmap='Greens')
plt.title('Target Amplitude in FT')

# Resulting amplitude in near-field
plt.subplot(1, 4, 4)
plt.imshow(np.abs(fftshift(fft2(E_nf_updated))), cmap='Greens')
plt.title('Resulting Amplitude in NF')

plt.show()

# Iterating
num_iterations = 100

for iteration in range(num_iterations):
    E_ft_updated = fft2(E_nf_updated)
    E_ft_updated = A_target * np.exp(1j * np.angle(E_ft_updated))
    E_nf_updated = ifft2(E_ft_updated)
    E_nf_updated = A_nf * np.exp(1j * np.angle(E_nf_updated))

    # Plotting intermediate results
    if iteration % 20 == 0:
        plt.imshow(np.abs(fftshift(fft2(E_nf_updated))), cmap='Greens')
        plt.title(f'Iteration {iteration}')
        plt.show()


# Display the final result
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(np.abs(E_ft_updated), cmap='Greens')
plt.title('Final Result in FF')

plt.subplot(1, 2, 2)
plt.imshow(np.abs(fftshift(fft2(E_nf_updated))), cmap='Greens')
plt.title('Final Result in NF')

plt.show()
