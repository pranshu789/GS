import numpy as np
import matplotlib.pyplot as plt
from zernike import RZern
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase

# Calibration factor
calibration_pixel_to_mm = 0.60 / 1440


# Load data
loaded_data = np.load('/Users/pranshudave/Desktop/Results/E_nf_updated.npz')
E_nf_updated = loaded_data['E_nf_updated']

E_nf_updated = (E_nf_updated[270:650,550:900])


# Image size and grids
image_size_x = len(E_nf_updated[0])
image_size_y = len(E_nf_updated)
x = np.linspace(0, calibration_pixel_to_mm * image_size_x, image_size_x)
y = np.linspace(0, calibration_pixel_to_mm * image_size_y, image_size_y)
X, Y = np.meshgrid(x - x.mean(), y - y.mean())


# Fit Zernike polynomials to the phase of E_nf_updated
zern_order = 6  # radial order
cart = RZern(zern_order)

# Convert the phase of E_nf_updated to radians
phase_rad = np.angle(E_nf_updated)

# Ensure phase_rad has the same shape as X and Y
phase_rad = phase_rad[:image_size_y, :image_size_x]

E_nf_angle = unwrap_phase((phase_rad))

phase_rad1 = E_nf_angle
# Create Cartesian grid
cart.make_cart_grid(X, Y)

# Fit Zernike polynomials to the phase data
coefficients = cart.fit_cart_grid(phase_rad1)[0]

# Plot individual Zernike polynomials
plt.figure(figsize=(10, 10))
for i in range(1, 10):
    plt.subplot(3, 3, i)

    # Create a coefficient vector with only one non-zero element
    c = np.zeros(cart.nk)
    c[i - 1] = 1.0

    # Evaluate Zernike polynomial on the grid
    Phi = cart.eval_grid(c, matrix=True)
    weighted_Phi = coefficients[i - 1] * Phi

    plt.imshow(weighted_Phi, origin='lower', extent=(X.min(), X.max(), Y.min(), Y.max()))
    plt.title(f'Zernike {i}')
    plt.axis('off')

desktop_path_zern= "/Users/pranshudave/Desktop/Results"
figure_path_zern = desktop_path_zern + "/Zernike3.png"

plt.savefig(figure_path_zern, bbox_inches='tight')
plt.show()

# Plotting
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.imshow(phase_rad, cmap='coolwarm', origin='lower', extent=(X.min(), X.max(), Y.min(), Y.max()))
plt.title('Original Phase')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(E_nf_angle, cmap='coolwarm', origin='lower', extent=(X.min(), X.max(), Y.min(), Y.max()))
plt.title('Unwrapped Phase')
plt.colorbar()

plt.subplot(2, 2, 3)
# Use the same grid for evaluating Zernike polynomials
Phi = cart.eval_grid(coefficients, matrix=True)
plt.imshow(Phi, cmap='viridis', origin='lower', extent=(X.min(), X.max(), Y.min(), Y.max()))
plt.title('Zernike Polynomials')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.plot(range(1, cart.nk + 1), coefficients, marker='.')
plt.title('Zernike Coefficients')

desktop_path_zern= "/Users/pranshudave/Desktop/Results"
figure_path_zern = desktop_path_zern + "/Zernike2.png"

plt.savefig(figure_path_zern, bbox_inches='tight')

plt.tight_layout()
plt.show()

print(len(coefficients))