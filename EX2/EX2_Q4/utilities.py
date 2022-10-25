import numpy as np
import cv2 as cv

base_images_address = 'images/'
base_results_address = 'results/'


def gaussian_mask(sigma, height, width, mu=0.0):
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))

    dist = np.sqrt(x * x + y * y)
    mask = np.exp(-((dist - mu) ** 2 / (2.0 * sigma ** 2)))

    return mask


def save_fft_channel_images(near_image_shifted_fft, far_image_shifted_fft, channel_name):
    amp_near_image_shifted_fft = np.abs(near_image_shifted_fft)
    amp_far_image_shifted_fft = np.abs(far_image_shifted_fft)

    log_amp_near_image_shifted_fft = np.log(amp_near_image_shifted_fft)
    log_amp_far_image_shifted_fft = np.log(amp_far_image_shifted_fft)

    scaled_log_amp_near_image_shifted_fft = log_amp_near_image_shifted_fft / log_amp_near_image_shifted_fft.max() * 255
    scaled_log_amp_far_image_shifted_fft = log_amp_far_image_shifted_fft / log_amp_far_image_shifted_fft.max() * 255

    cv.imwrite(base_images_address + "q4_05_dft_near_" + channel_name + ".jpg", scaled_log_amp_near_image_shifted_fft)
    cv.imwrite(base_images_address + "q4_06_dft_far_" + channel_name + ".jpg", scaled_log_amp_far_image_shifted_fft)


def save_filters(near_filter, far_filter, r, s):
    near_filter = near_filter * 255
    far_filter = far_filter * 255

    cv.imwrite(base_results_address + "Q4_07_highpass_" + str(r) + ".jpg", near_filter)
    cv.imwrite(base_results_address + "Q4_08_lowpass_" + str(s) + ".jpg", far_filter)


def apply_circle_mask(src_image, radius, inverse):
    h, w = src_image.shape[:2]
    mask = create_circular_mask(h, w, radius=radius)
    masked_img = src_image.copy()
    if inverse:
        masked_img[mask] = 0
    else:
        masked_img[~mask] = 0

    return masked_img


def save_filters_cutoff(near_filter, far_filter, near_radius, far_radius):
    near_image_filter_cutoff = apply_circle_mask(near_filter, near_radius, True)
    far_image_filter_cutoff = apply_circle_mask(far_filter, far_radius, False)

    near1 = apply_circle_mask(near_filter, near_filter, False)
    near2 = apply_circle_mask(near1, far_radius, True)

    far1 = apply_circle_mask(far_filter, near_radius, False)
    far2 = apply_circle_mask(far1, far_filter, True)

    ave = .5 * far2 + .5 * near2

    near_image_filter_cutoff = near_image_filter_cutoff + ave
    far_image_filter_cutoff = far_image_filter_cutoff + ave

    cv.imwrite(base_results_address + "Q4_09_highpass_cutoff.jpg", near_image_filter_cutoff * 255)
    cv.imwrite(base_results_address + "Q4_10_lowpass_cutoff.jpg", far_image_filter_cutoff * 255)

    return near_image_filter_cutoff, far_image_filter_cutoff


def create_hybrid(far_image, near_image, channel_name):
    near_image_fft = np.fft.fft2(near_image)
    far_image_fft = np.fft.fft2(far_image)

    near_image_shifted_fft = np.fft.fftshift(near_image_fft)
    far_image_shifted_fft = np.fft.fftshift(far_image_fft)

    save_fft_channel_images(near_image_shifted_fft, far_image_shifted_fft, channel_name)

    rows, cols = near_image.shape

    near_image_filter = 1 - gaussian_mask(0.04, rows, cols)
    far_image_filter = gaussian_mask(0.07, rows, cols)

    save_filters(near_image_filter, far_image_filter, 0.04, 0.07)
    near_image_filter, far_image_filter = save_filters_cutoff(near_image_filter, far_image_filter, 10, 25)

    fil_near_image_fft = near_image_shifted_fft * near_image_filter
    fil_far_image_fft = far_image_shifted_fft * far_image_filter

    # fil_near_result_image_ishifted = np.fft.ifftshift(fil_near_image_fft)
    # fil_far_result_image_ishifted = np.fft.ifftshift(fil_far_image_fft)

    fil_result_image_fft = fil_near_image_fft + fil_far_image_fft

    fil_result_image_ishifted = np.fft.ifftshift(fil_result_image_fft)
    fil_result_image = np.fft.ifft2(fil_result_image_ishifted)

    return np.real(fil_result_image), fil_result_image_fft, fil_near_image_fft, fil_far_image_fft


def merge_fft_channel_images():
    near_fft_b = cv.imread(base_images_address + "q4_05_dft_near_b.jpg", 0)
    near_fft_g = cv.imread(base_images_address + "q4_05_dft_near_g.jpg", 0)
    near_fft_r = cv.imread(base_images_address + "q4_05_dft_near_r.jpg", 0)

    far_fft_b = cv.imread(base_images_address + "q4_06_dft_far_b.jpg", 0)
    far_fft_g = cv.imread(base_images_address + "q4_06_dft_far_b.jpg", 0)
    far_fft_r = cv.imread(base_images_address + "q4_06_dft_far_b.jpg", 0)

    near_fft = cv.merge((near_fft_b, near_fft_g, near_fft_r))
    far_fft = cv.merge((far_fft_b, far_fft_g, far_fft_r))

    cv.imwrite(base_results_address + "q4_05_dft_near.jpg", near_fft)
    cv.imwrite(base_results_address + "q4_06_dft_far.jpg", far_fft)


def save_fft_channel_images(near_image_shifted_fft, far_image_shifted_fft, channel_name):
    amp_near_image_shifted_fft = np.abs(near_image_shifted_fft)
    amp_far_image_shifted_fft = np.abs(far_image_shifted_fft)

    log_amp_near_image_shifted_fft = np.log(amp_near_image_shifted_fft)
    log_amp_far_image_shifted_fft = np.log(amp_far_image_shifted_fft)

    scaled_log_amp_near_image_shifted_fft = log_amp_near_image_shifted_fft / log_amp_near_image_shifted_fft.max() * 255
    scaled_log_amp_far_image_shifted_fft = log_amp_far_image_shifted_fft / log_amp_far_image_shifted_fft.max() * 255

    cv.imwrite(base_images_address + "q4_05_dft_near_" + channel_name + ".jpg", scaled_log_amp_near_image_shifted_fft)
    cv.imwrite(base_images_address + "q4_06_dft_far_" + channel_name + ".jpg", scaled_log_amp_far_image_shifted_fft)


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def cal_scaled_amp(b, g, r):

    b = np.abs(b)
    g = np.abs(g)
    r = np.abs(r)

    b = np.log(b + 0.1)
    g = np.log(g + 0.1)
    r = np.log(r + 0.1)

    b = b / b.max() * 255
    g = g / g.max() * 255
    r = r / r.max() * 255

    return cv.merge((b, g, r))
