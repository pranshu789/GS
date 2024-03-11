import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fresnel_diffraction(x, y, E, z, wavelength):
    x_scale = x[-1] - x[0]
    y_scale = y[-1] - y[0]

    Ny, Nx = E.shape
    factor = 1 / np.pi

    F_E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E)))
    kx_vec = np.pi * (1 / (x_scale / Nx)) * np.arange(-Nx/2, Nx/2) / Nx
    ky_vec = np.pi * (1 / (y_scale / Ny)) * np.arange(-Ny/2, Ny/2) / Ny
    k_x, k_y = np.meshgrid(kx_vec, ky_vec)

    quadratic_phase = -(1j * (wavelength * z * factor) * (k_x**2 + k_y**2))
    F_E_D = F_E * np.exp(quadratic_phase)

    E_D = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(F_E_D)))
    return E_D

def normalize_and_compare(A, B):
    # Normalize A
    A_norm = (A - np.min(A)) / (np.max(A) - np.min(A))

    # Normalize B
    B_norm = (B - np.min(B)) / (np.max(B) - np.min(B))

    # Calculate C and error
    C = A_norm - B_norm
    C_out = A_norm**2 - B_norm**2
    C_sq = C**2
    A_sq = A_norm**2

    error = np.sum(C_sq) / np.sum(A_sq)

    return C_out, error

def gaussian(x, y, sigma):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


def calculate_fwhm(x, y):
    half_max = max(y) / 2.0

    # Fit the data to a Gaussian function
    popt, _ = curve_fit(gaussian, x, y, p0=[max(y), np.std(x)])

    # Extract parameters of the fitted Gaussian
    A, sigma = popt

    # Calculate FWHM using the standard deviation (sigma) of the Gaussian
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    return fwhm


# Set parameters (all in mm)
N = M = 256
calibration = 0.01
wavelength = 0.000535
z_value = 100.0
num_iterations = 100
sigma = 0.33

# Create grid
x = np.arange(1, M + 1) * calibration
y = np.arange(1, N + 1) * calibration
X, Y = np.meshgrid(x - x.mean(), y - y.mean())


#def test_gaussian():
# Initial Gaussian field
initial_gaussian = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
initial_field = initial_gaussian / np.max(initial_gaussian)

# Propagate Gaussian with flat phase to +z
propagated_field = fresnel_diffraction(x, y, initial_field, z_value, wavelength)

# Plot initial and propagated fields
central_row_initial = np.abs(initial_field[len(x) // 2, :])
central_row_propagated = np.abs(propagated_field[len(x) // 2, :])

plt.figure(figsize=(10, 5))
plt.plot(x, central_row_initial, label='Initial Gaussian')
plt.plot(x, central_row_propagated, label='Propagated Gaussian')
plt.title('Evolution of Gaussian Field through Fresnel Propagation')
plt.xlabel('Position (meters)')
plt.ylabel('Intensity')
plt.legend()
plt.show()

original_fwhm_comparison = calculate_fwhm(x, central_row_initial)
propagated_fwhm_comparison = calculate_fwhm(x, central_row_propagated)

# Calculating the theoretical FWHM at +z using the beam waist equation
w_0_comparison = 0.849 * original_fwhm_comparison
theoretical_fwhm_comparison = 1.178 * w_0_comparison * np.sqrt(1 + ((wavelength * z_value) / (np.pi * (w_0_comparison ** 2))) ** 2)

# Print results
print(f"Original Gaussian FWHM: {original_fwhm_comparison}")
print(f"Propagated Gaussian FWHM: {propagated_fwhm_comparison}")
print(f"Theoretical FWHM at +z: {theoretical_fwhm_comparison}")



#def create_target():
A_nf = gaussian(X, Y, sigma)

# phase pattern
phi_nf = np.zeros_like(A_nf)

# quadrants with 0, π, 0, π phase values
phi_nf[:N // 2, :N // 2] = 0  # top-left
phi_nf[:N // 2, N // 2:] = np.pi/2   # top-right
phi_nf[N // 2:, :N // 2] = np.pi/2   # bottom-left
phi_nf[N // 2:, N // 2:] = 0  # bottom-right

E_nf1 = A_nf * np.exp(1j * phi_nf)

E_final = fresnel_diffraction(x, y, E_nf1, z_value, wavelength)

# Initialize the array to store MSE values over iterations
mse_values = [1]

# GS Algorithm
#def GS_Algorithm(E_final):
A_nf = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
phi_start = np.ones_like(A_nf)  # Unity phase
E_nf = A_nf * np.exp(1j * phi_start)

# Propagate the field forward to the desired plane
E_ff_updated = fresnel_diffraction(x, y, E_nf, z_value, wavelength)

# Fresnel propagation forward and backward for GS iterations
for iteration in range(num_iterations):
    E_ff_updated = np.abs(E_final) * np.exp(1j * np.angle(E_ff_updated))  # Update the phase using the target amplitude
    E_nf_updated = fresnel_diffraction(x, y, E_ff_updated, -z_value, wavelength)  # Propagate the field back to the near field
    E_nf_updated = np.abs(E_nf) * np.exp(1j * np.angle(E_nf_updated))  # Update the amplitude using the amplitude of the retrieved field
    E_ff_updated = fresnel_diffraction(x, y, E_nf_updated, z_value, wavelength)  # Propagate the field forward to the desired plane

    if iteration % 20 == 0:
        plt.imshow(np.angle(E_nf_updated), cmap='Greens')
        plt.title(f'Iteration {iteration}')
        plt.colorbar()
        plt.show()
    # Calculate absolute differences between E_final and E_ff_updated
    diff, mse = normalize_and_compare(np.abs(E_ff_updated), np.abs(E_final))
    mse_values.append(mse)

# Update the phase using the target amplitude

E_f1_updated = np.abs(E_final) * np.exp(1j * np.angle(E_ff_updated))

# Propagate the field back to the near field
E_nf_updated = fresnel_diffraction(x, y, E_f1_updated, -z_value, wavelength)
# Display results after GS iterations

plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20, 12))
plt.suptitle('Simulation of GS-Fresnel with quadrant-phase', fontsize=20)

# Display the initial target field
plt.subplot(2, 3, 3)
plt.imshow(np.abs(E_final), cmap='Greens', extent=[X.min(), X.max(), Y.min(), Y.max()])
plt.title('Created Propagated Field')
plt.colorbar()


# Display the initial phase
plt.subplot(2, 3, 2)
plt.imshow(np.angle(E_nf), cmap='Greens', extent=[X.min(), X.max(), Y.min(), Y.max()])
plt.title('Initialized Focus Field for GS')
plt.colorbar()

# Display the initial near field
plt.subplot(2, 3, 1)
plt.imshow(np.abs(E_nf), cmap='Greens', extent=[X.min(), X.max(), Y.min(), Y.max()])
plt.title('Created Focus Field')
plt.colorbar()

# Display the initial field with phase
plt.subplot(2, 3, 4)
plt.imshow(np.angle(E_nf1), cmap='seismic', vmin=-np.pi, vmax=np.pi, extent=[X.min(), X.max(), Y.min(), Y.max()])
plt.title('Created Focus Phase')
plt.colorbar()

# Display the retrieved phase
plt.subplot(2, 3, 5)
plt.imshow(np.angle(E_nf_updated), cmap='seismic', vmin=-np.pi, vmax=np.pi, extent=[X.min(), X.max(), Y.min(), Y.max()])
plt.title('Retrieved Phase - after GS')
plt.colorbar()

# Display the final result
plt.subplot(2, 3, 6)
plt.imshow(np.abs(E_ff_updated), cmap='Greens', extent=[X.min(), X.max(), Y.min(), Y.max()])
plt.title('Retrieved Propagated Field - after GS')
plt.colorbar()



desktop_path_test= "/Users/pranshudave/Desktop/Results"
figure_path_test = desktop_path_test + "/test.png"

plt.savefig(figure_path_test, bbox_inches='tight')

plt.show()


# Plot the absolute differences
plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.imshow(np.abs(diff), cmap='viridis', extent=[X.min(), X.max(), Y.min(), Y.max()])
plt.title('Absolute Differences')
plt.colorbar()

# Plot the convergence of MSE over iterations
plt.subplot(1, 2, 2)
plt.plot(range(0, num_iterations+1), np.log(mse_values), marker='o')
plt.title('Convergence of Error value over Iterations')
plt.xlabel('Iteration #')
plt.ylabel('Log(Rel. error) (unitless)')
plt.grid(True)

# Save the figure with absolute differences to the desktop
desktop_path_conv = "/Users/pranshudave/Desktop/Results"
figure_path_conv = desktop_path_conv + "/test-convergence.png"

plt.savefig(figure_path_conv, bbox_inches='tight')
plt.show()

