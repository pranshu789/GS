import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifftshift
from skimage import data, img_as_float, color, exposure
from skimage.restoration import unwrap_phase


def propagate_near_to_far(near_field, wavelength, aperture_diameter, focal_length):
    # defining constants
    k = 2 * np.pi / wavelength

    # creating coordinate grids
    x, y = np.meshgrid(np.linspace(-aperture_diameter / 2, aperture_diameter / 2, near_field.shape[1]),
                       np.linspace(-aperture_diameter / 2, aperture_diameter / 2, near_field.shape[0]))

    # performing Fourier transform to get the far-field complex field
    far_field_complex = np.fft.fftshift(fft2(ifftshift(near_field)))

    # calculating the amplitude and phase
    far_field_intensity = np.abs(far_field_complex)
    far_field_phase = np.angle(far_field_complex)


    return far_field_intensity, far_field_phase


# Load near-field phase from the provided data
data = np.load('/Users/pranshudave/Desktop/Results/E_nf_updated.npz')
near_field = data['E_nf_updated']


padded_near_field = np.zeros((1300,1300))

# Specify the region of interest
roi = near_field[425:600, 625:775]

# Assign the values of the region of interest to the corresponding part in the padded_near_field
padded_near_field[540:715, 575:725] = roi

near_field=padded_near_field

# Setting parameters (all in mm)
calibration_pixel_to_mm = 0.60 / 1440  # Conversion factor from pixels to mm
wavelength = 0.000535
focal_length = 100.0  # Focal length of the lens in mm
size = calibration_pixel_to_mm

# Image size and grids
image_size_x = len(near_field[0])
image_size_y = len(near_field)
x = np.linspace(0, calibration_pixel_to_mm * image_size_x, image_size_x)
y = np.linspace(0, calibration_pixel_to_mm * image_size_y, image_size_y)
X, Y = np.meshgrid(x - x.mean(), y - y.mean())

# Propagation parameters
aperture_diameter = 1.0

# Propagate near field to far field
far_field_intensity, far_field_phase = propagate_near_to_far(near_field, wavelength, aperture_diameter,
                                                           focal_length)

far_field_intensity = far_field_intensity[625:675,625:675]
far_field_phase = far_field_phase[625:675,625:675]
#far_field_phase = unwrap_phase((far_field_phase))

# normalized image
far_field_intensity = far_field_intensity**2
max_pixel_value = np.max(far_field_intensity) #4095
far_field_intensity = far_field_intensity / max_pixel_value

# logarithmic scale with a small constant added to avoid log(0)
small_constant = 1e-10

plt.figure(figsize=(12, 6))
plt.suptitle("Far-field Fraunhofer Approximation for Gaussian", fontsize=25)

plt.subplot(1, 3, 1)
plt.imshow((np.abs(near_field) + small_constant), cmap='plasma')
plt.title('Near Field')
#plt.axis([650, 800, 400, 500])
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow((far_field_intensity + small_constant)**2, cmap='plasma')
plt.title('Far Field Intensity')
#plt.axis([650, 800, 450, 650])
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(far_field_phase, cmap='coolwarm')
plt.title('Far Field Phase')
plt.colorbar()

desktop_path_conv = "/Users/pranshudave/Desktop/Results"
figure_path_conv = desktop_path_conv + "/far-fieldg.png"

plt.savefig(figure_path_conv, bbox_inches='tight')

plt.show()

np.savez('/Users/pranshudave/Desktop/Results/E_farfield.npz', E_farfield=far_field_phase)

