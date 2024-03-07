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
wavelength = 0.0006328
z_value = 100.0
num_iterations = 100
sigma = 0.4

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

# phase pattern
phi_nf = np.zeros_like(A_nf)

center_x, center_y = phi_nf.shape[0] // 2, phi_nf.shape[1] // 2
square_size = 15  # Adjust the size of the square as needed

phi_nf[center_x - square_size//2:center_x + square_size//2,
       center_y - square_size//2:center_y + square_size//2] = 1  # Assign the desired phase value


E_nf1 = phi_nf
plt.imshow(np.abs(E_nf1), cmap='PuRd')
plt.colorbar()
plt.show()
E_ft = np.fft.fftshift(fftpack.fft2(E_nf1))
plt.imshow(np.abs(E_ft), cmap='PuRd')
plt.colorbar()
plt.show()





# GS Algorithm
#def GS_Algorithm(E_final):
A_nf = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
phi_start = np.ones_like(A_nf)  # Unity phase
E_nf = A_nf * np.exp(1j * phi_start)

# Propagate the field forward to the desired plane
E_ff_updated = fresnel_diffraction(x, y, E_nf, z_value, wavelength)
plt.imshow(np.angle(E_ff_updated), cmap='twilight_shifted', vmin=0, vmax=2*np.pi)
plt.title('propagated target phase')
plt.colorbar()
plt.show()
plt.imshow(np.abs(E_ff_updated), cmap='Greens')
plt.title('propagated target amplitude')
plt.colorbar()
plt.show()

# Fresnel propagation forward and backward for GS iterations
for iteration in range(num_iterations):
    E_ff_updated = np.abs(E_final) * np.exp(1j * np.angle(E_ff_updated))  # Update the phase using the target amplitude
    if iteration % 100 == 0:
        plt.imshow(np.angle(E_ff_updated), cmap='twilight_shifted', vmin=0, vmax=2*np.pi)
        plt.title('updated target phase')
        plt.colorbar()
        plt.show()
    if iteration % 100 == 0:
        plt.imshow(np.abs(E_ff_updated), cmap='Greens')
        plt.title('updated target amplitude')
        plt.colorbar()
        plt.show()
    E_nf_updated = fresnel_diffraction(x, y, E_ff_updated, -z_value, wavelength)  # Propagate the field back to the near field
    if iteration % 1000 == 0:
        plt.imshow(np.angle(E_nf_updated), cmap='twilight_shifted', vmin=0, vmax=2*np.pi)
        plt.title('propagated source phase')
        plt.colorbar()
        plt.show()
    if iteration % 100 == 0:
        plt.imshow(np.abs(E_nf_updated), cmap='Greens')
        plt.title('propagated source amplitude')
        plt.colorbar()
        plt.show()
    E_nf_updated = np.abs(E_nf) * np.exp(1j * np.angle(E_nf_updated))  # Update the amplitude using the amplitude of the retrieved field
    if iteration % 100 == 0:
        plt.imshow(np.angle(E_nf_updated), cmap='twilight_shifted', vmin=0, vmax=2*np.pi)
        plt.title('uPdated source phase')
        plt.colorbar()
        plt.show()
    if iteration % 100 == 0:
        plt.imshow(np.abs(E_nf_updated), cmap='Greens')
        plt.title('Updated source amplitude')
        plt.colorbar()
        plt.show()
    E_ff_updated = fresnel_diffraction(x, y, E_nf_updated, z_value, wavelength)  # Propagate the field forward to the desired plane
    if iteration % 100 == 0:
        plt.imshow(np.angle(E_ff_updated), cmap='Greens')
        plt.title('propagated target phase')
        plt.colorbar()
        plt.show()
    if iteration % 100 == 0:
        plt.imshow(np.abs(E_ff_updated), cmap='Greens')
        plt.title('propagated target amplitude')
        plt.colorbar()
        plt.show()
    #if iteration % 20 == 0:
        #plt.imshow(np.angle(E_nf_updated), cmap='Greens')
        #plt.title(f'Iteration {iteration}')
        #plt.colorbar()
        #plt.show()


# Update the phase using the target amplitude

E_f1_updated = np.abs(E_final) * np.exp(1j * np.angle(E_ff_updated))

# Propagate the field back to the near field
E_nf_updated = fresnel_diffraction(x, y, E_f1_updated, -z_value, wavelength)
# Display results after GS iterations
plt.figure(figsize=(20, 12))

# Display the initial target field
plt.subplot(2, 3, 3)
plt.imshow(np.abs(E_final), cmap='Greens')
plt.title('Target Field')
plt.colorbar()

# Display the initial phase
plt.subplot(2, 3, 2)
plt.imshow(np.angle(E_nf), cmap='twilight_shifted', vmin=0, vmax=2*np.pi)
plt.title('Initial Phase')
plt.colorbar()

# Display the initial near field
plt.subplot(2, 3, 1)
plt.imshow(np.abs(E_nf), cmap='Greens')
plt.title('Initial Field')
plt.colorbar()

# Display the initial field with phase
plt.subplot(2, 3, 4)
plt.imshow(np.angle(E_nf1), cmap='twilight_shifted', vmin=0, vmax=2*np.pi)
plt.title('Initial Target Phase')
plt.colorbar()

# Display the retrieved phase
plt.subplot(2, 3, 5)
phase_unwrapped = np.mod(np.angle(E_nf_updated), 2*np.pi)
plt.imshow(phase_unwrapped, cmap='twilight_shifted', vmin=0, vmax=2*np.pi)
#plt.imshow(np.angle(E_nf_updated), cmap='seismic')
plt.title('Retrieved Phase - after GS')
plt.colorbar()

# Display the final result
plt.subplot(2, 3, 6)
plt.imshow(np.abs(E_ff_updated), cmap='Greens')
plt.title('Retrieved Final Field - after GS')
plt.colorbar()

desktop_path_test= "/Users/pranshudave/Desktop/Results"
figure_path_test = desktop_path_test + "/test.png"

plt.savefig(figure_path_test, bbox_inches='tight')

plt.show()

