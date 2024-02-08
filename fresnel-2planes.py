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

def normalize_and_calibrate(image, exposure_time):
    # for 12-bit monochrome scale
    max_pixel_value = np.max(image) # 4095

    # normalized image
    normalized_image = image / max_pixel_value

    # calibrated based on exposure time
    calibrated_image = normalized_image #* exposure_time

    return calibrated_image

def load_intensity_image(file_path):
    # Load intensity image from file_path
    intensity_image = plt.imread(file_path)
    return intensity_image

#loading images for E_nf and E_final
E_nf_intensity = load_intensity_image('/Users/pranshudave/Desktop/Set2/default1.tiff')
E_final_intensity = load_intensity_image('/Users/pranshudave/Desktop/Set2/offminus1.tiff')

#integration times
exposure_time_E_nf = 750
exposure_time_E_final = 40000

# normalizing and calibrate images
E_nf_intensity = normalize_and_calibrate(E_nf_intensity, exposure_time_E_nf)
E_final_intensity = normalize_and_calibrate(E_final_intensity, exposure_time_E_final)

# setting parameters (all in mm)
calibration_pixel_to_mm = 0.60 / 1440  # Conversion factor from pixels to mm
wavelength = 0.0006328
z_value = -5.00
num_iterations = 100
#sigma = 0.3

#image size and grids
image_size_x = 1440
image_size_y = 1080
x = np.linspace(0, calibration_pixel_to_mm * image_size_x, image_size_x)
y = np.linspace(0, calibration_pixel_to_mm * image_size_y, image_size_y)
X, Y = np.meshgrid(x - x.mean(), y - y.mean())

# Initial fields
E_nf = np.sqrt(E_nf_intensity) * np.exp(1j * np.zeros_like(E_nf_intensity))  # Default phase: zero
E_final = np.sqrt(E_final_intensity) * np.exp(1j * np.zeros_like(E_final_intensity))  # Default phase: zero


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


plt.figure(figsize=(20, 12))

# Display the retrieved phase
plt.subplot(2, 3, 5)
plt.imshow(np.angle(E_ff_updated), cmap='coolwarm')
plt.title('Retrieved Phase - Far Field')
plt.colorbar()

# Display the final result
plt.subplot(2, 3, 6)
plt.imshow(np.abs(E_ff_updated), cmap='plasma')
plt.title('Final target Field')
plt.colorbar()


# Update the phase using the target amplitude
E_ff_updated = np.abs(E_final) * np.exp(1j * np.angle(E_ff_updated))

# Propagate the field back to the near field
E_nf_updated = fresnel_diffraction(x, y, E_ff_updated, -z_value, wavelength)
# Display results after GS iterations


# Display the initial target field
plt.subplot(2, 3, 3)
plt.imshow(np.abs(E_final), cmap='plasma')
plt.title('Far Field')
plt.colorbar()

# Display the initial phase
plt.subplot(2, 3, 2)
plt.imshow(np.angle(E_nf_updated), cmap='coolwarm')
plt.title('Retrieved Phase - Near Field')
plt.colorbar()

# Display the initial near field
plt.subplot(2, 3, 1)
plt.imshow(np.abs(E_nf), cmap='plasma')
plt.title('Near Field')
plt.colorbar()

# Display the initial field with phase
plt.subplot(2, 3, 4)
plt.imshow(np.abs(E_nf_updated), cmap='plasma')
plt.title('Near Field Final')
plt.colorbar()

# Save the figure to the desktop
desktop_path = "/Users/pranshudave/Desktop/Results"
figure_path = desktop_path + "/normal-minus.png"

plt.savefig(figure_path, bbox_inches='tight')


plt.show()

