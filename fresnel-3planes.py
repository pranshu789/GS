import matplotlib.pyplot as plt
import numpy as np
import time

import numpy as np
import matplotlib.pyplot as plt

def fresnel_diffraction(x, y, E, z, wavelength):
    x_scale = x[-1] - x[0]
    y_scale = y[-1] - y[0]

    Ny, Nx = E.shape
    factor = 1 / np.pi

    F_E = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E)))
    kx_vec = np.pi * (1 / (x_scale / Nx)) * np.arange(-Nx / 2, Nx / 2) / Nx
    ky_vec = np.pi * (1 / (y_scale / Ny)) * np.arange(-Ny / 2, Ny / 2) / Ny
    k_x, k_y = np.meshgrid(kx_vec, ky_vec)

    quadratic_phase = -(1j * (wavelength * z * factor) * (k_x ** 2 + k_y ** 2))
    F_E_D = F_E * np.exp(quadratic_phase)

    E_D = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(F_E_D)))
    return E_D

def gaussian(x, y, sigma):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

def normalize_and_calibrate(image, exposure_time):
    # for 12-bit monochrome scale
    max_pixel_value = np.max(image) #4095

    # normalized image
    normalized_image = image / max_pixel_value

    # calibrated based on exposure time
    calibrated_image = normalized_image #* exposure_time

    return calibrated_image

def load_intensity_image(file_path):
    # Load intensity image from file_path
    intensity_image = plt.imread(file_path)
    return intensity_image

def calculate_mse(image1, image2):
    return np.mean((image1 - image2)**2)

# Initialize arrays to store error values
mse_values = []

# loading images for E_nf, E_final_off_minus, and E_final_off_plus
E_nf_intensity = load_intensity_image('/Users/pranshudave/Desktop/SET1/default1.tiff')
E_final_off_minus_intensity = load_intensity_image('/Users/pranshudave/Desktop/SET1/offminus1.tiff')
E_final_off_plus_intensity = load_intensity_image('/Users/pranshudave/Desktop/SET1/offplus1.tiff')

# integration times
exposure_time_E_nf = 750
exposure_time_E_final_off_minus = 40000
exposure_time_E_final_off_plus = 50000

# normalizing and calibrate images
E_nf_intensity = normalize_and_calibrate(E_nf_intensity, exposure_time_E_nf)
E_final_off_minus_intensity = normalize_and_calibrate(E_final_off_minus_intensity, exposure_time_E_final_off_minus)
E_final_off_plus_intensity = normalize_and_calibrate(E_final_off_plus_intensity, exposure_time_E_final_off_plus)


# Setting parameters (all in mm)
calibration_pixel_to_mm = 0.60 / 1440  # Conversion factor from pixels to mm
wavelength = 0.0006328
z1 = 5.0
z2 = 5.0  # Distances of three planes
num_iterations = 100

# Image size and grids
image_size_x = 1440
image_size_y = 1080
x = np.linspace(0, calibration_pixel_to_mm * image_size_x, image_size_x)
y = np.linspace(0, calibration_pixel_to_mm * image_size_y, image_size_y)
X, Y = np.meshgrid(x - x.mean(), y - y.mean())

E_nf = np.sqrt(E_nf_intensity) * np.exp(1j * np.zeros_like(E_nf_intensity))  # Default phase: zero
E_final_off_minus = np.sqrt(E_final_off_minus_intensity) * np.exp(1j * np.zeros_like(E_final_off_minus_intensity))  # Default phase: zero
E_final_off_plus = np.sqrt(E_final_off_plus_intensity) * np.exp(1j * np.zeros_like(E_final_off_plus_intensity))  # Default phase: zero

# Propagate the field forward to the off-minus plane
E_offm_updated = fresnel_diffraction(x, y, E_nf, -z1, wavelength)

# Fresnel propagation forward and backward for GS iterations
for iteration in range(num_iterations):
    # Propagate to off-minus plane
    E_offm_updated = np.abs(E_final_off_minus) * np.exp(1j * np.angle(E_offm_updated))

    # Propagate to default plane
    E_nf_updated = fresnel_diffraction(x, y, E_offm_updated, z1, wavelength)
    E_nf_updated = np.abs(E_nf) * np.exp(1j * np.angle(E_nf_updated))

    # Propagate to off-plus plane
    E_offp_updated = fresnel_diffraction(x, y, E_nf_updated, z2, wavelength)
    E_offp_updated = np.abs(E_final_off_plus) * np.exp(1j * np.angle(E_offp_updated))

    # Propagate back to default plane
    E_nf_updated = fresnel_diffraction(x, y, E_offp_updated, -z2, wavelength)
    E_nf_updated = np.abs(E_nf) * np.exp(1j * np.angle(E_nf_updated))

    # Propagate back to off-minus plane
    E_offm_updated = fresnel_diffraction(x, y, E_nf_updated, -z1, wavelength)

    # Calculate MSE and store in the array
    mse_value = calculate_mse(np.abs(E_final_off_minus), np.abs(E_offm_updated))
    mse_values.append(mse_value)

    if iteration % 50 == 0:
        plt.imshow(np.angle(E_nf_updated), cmap='Greens')
        plt.title(f'Iteration {iteration}')
        plt.colorbar()
        plt.show()

# Display and save the final results
plt.figure(figsize=(24, 20))
plt.suptitle('Gerchberg-Saxton Algorithm - Three Planes - Normal', fontsize=40)

# Display the retrieved phase
plt.subplot(3, 3, 1)
plt.imshow(np.abs(normalize_and_calibrate(E_final_off_minus,1)), cmap='plasma')
plt.title('Actual Off-Minus Field')
plt.colorbar()

plt.subplot(3, 3, 2)
plt.imshow(np.angle(E_offm_updated), cmap='coolwarm')
plt.title('Retrieved Off-Minus Phase')
plt.colorbar()

# Display the final result
plt.subplot(3, 3, 3)
plt.imshow(np.abs(normalize_and_calibrate(E_offm_updated,1)), cmap='plasma')
plt.title('Retrieved Off-Minus Field')
plt.colorbar()

diff_off_minus = np.abs(np.abs(normalize_and_calibrate(E_final_off_minus,1)) - np.abs(normalize_and_calibrate(E_offm_updated,1)))


# Propagate to default plane
E_offm_updated = np.abs(E_final_off_minus) * np.exp(1j * np.angle(E_offm_updated))
E_nf_updated = fresnel_diffraction(x, y, E_offm_updated, z1, wavelength)

# Display the retrieved phase
plt.subplot(3, 3, 4)
plt.imshow(np.abs(E_nf), cmap='plasma')
plt.title('Actual Focus Field')
plt.colorbar()

plt.subplot(3, 3, 5)
plt.imshow(np.angle(E_nf_updated), cmap='coolwarm')
plt.title('Retrieved Focus Phase')
plt.colorbar()

# Display the final result
plt.subplot(3, 3, 6)
plt.imshow(np.abs(normalize_and_calibrate(E_nf_updated,1)), cmap='plasma')
plt.title('Retrieved Focus Field')
plt.colorbar()

diff_focus = np.abs(np.abs(normalize_and_calibrate(E_nf,1)) - np.abs(normalize_and_calibrate(E_nf_updated,1)))

# Propagate to off-plus plane
E_nf_updated = np.abs(E_nf) * np.exp(1j * np.angle(E_nf_updated))
E_offp_updated = fresnel_diffraction(x, y, E_nf_updated, z2, wavelength)

# Display the retrieved phase
plt.subplot(3, 3, 7)
plt.imshow(np.abs(E_final_off_plus), cmap='plasma')
plt.title('Actual Off-Plus Field')
plt.colorbar()

plt.subplot(3, 3, 8)
plt.imshow(np.angle(E_offp_updated), cmap='coolwarm')
plt.title('Retrieved Off-Plus Phase')
plt.colorbar()

# Display the final result
plt.subplot(3, 3, 9)
plt.imshow(np.abs(normalize_and_calibrate(E_offp_updated,1)), cmap='plasma')
plt.title('Retrieved Off-Plus Field')
plt.colorbar()


# Save the figure to the desktop
desktop_path = "/Users/pranshudave/Desktop/Results"
figure_path = desktop_path + "/normal.png"

plt.savefig(figure_path, bbox_inches='tight')
plt.show()


# Calculate absolute differences between Actual and Retrieved fields
diff_off_plus = np.abs(np.abs(normalize_and_calibrate(E_final_off_plus,1)) - np.abs(normalize_and_calibrate(E_offp_updated,1)))

# Plotting the final absolute differences
plt.figure(figsize=(18, 8))
plt.suptitle('Gerchberg-Saxton Algorithm - Three Planes - Absolute Differences', fontsize=40)

# Display the absolute difference for Off-Minus plane
plt.subplot(1, 3, 1)
plt.imshow(diff_off_minus, cmap='plasma')
plt.title('Absolute Difference - Off-Minus Plane')
plt.colorbar()

# Display the absolute difference for Focus plane
plt.subplot(1, 3, 2)
plt.imshow(diff_focus, cmap='plasma')
plt.title('Absolute Difference - Focus Plane')
plt.colorbar()

# Display the absolute difference for Off-Plus plane
plt.subplot(1, 3, 3)
plt.imshow(diff_off_plus, cmap='plasma')
plt.title('Absolute Difference - Off-Plus Plane')
plt.colorbar()

# Save the figure with absolute differences to the desktop
desktop_path_diff = "/Users/pranshudave/Desktop/Results"
figure_path_diff = desktop_path_diff + "/normal-differences.png"

plt.savefig(figure_path_diff, bbox_inches='tight')
plt.show()


# Plotting the convergence of MSE over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_iterations + 1), mse_values, marker='o')
plt.title('Convergence of MSE over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error (MSE)')
plt.grid(True)
# Save the figure with absolute differences to the desktop
desktop_path_conv = "/Users/pranshudave/Desktop/Results"
figure_path_conv = desktop_path_conv + "/normal-convergence.png"

plt.savefig(figure_path_conv, bbox_inches='tight')
plt.show()