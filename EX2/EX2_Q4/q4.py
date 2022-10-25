from utilities import *
from match import *

far_image = cv.imread(base_images_address + "far.jpg")
near_image = cv.imread(base_images_address + "near.jpg")

old_near_image = near_image
near_image = match(near_image, far_image)

# far_blue, far_green, far_red
# near_blue, near_green, near_red
fb, fg, fr = cv.split(far_image)
nb, ng, nr = cv.split(near_image)

# result_spatial_domain_blue, result_frequency_domain_blue, near_result_frequency_blue, far_result_frequency_blue
# ....
rsb, rfb, nrfb, frfb = create_hybrid(fb, nb, 'b')
rsg, rfg, nrfg, frfg = create_hybrid(fg, ng, 'g')
rsr, rfr, nrfr, frfr = create_hybrid(fr, nr, 'r')

merge_fft_channel_images()

near_frequency_result = cal_scaled_amp(nrfb, nrfg, nrfr)
far_frequency_result = cal_scaled_amp(frfb, frfg, frfr)
frequency_result = cal_scaled_amp(rfb, rfg, rfr)

domain_result = cv.merge((rsb, rsg, rsr))

realized_domain_result = cv.resize(domain_result, (far_image.shape[1] // 4, far_image.shape[0] // 4),
                                   interpolation=cv.INTER_AREA)

cv.imwrite(base_results_address + "q4_01_near.jpg", old_near_image)
cv.imwrite(base_results_address + "q4_02_far.jpg", far_image)
cv.imwrite(base_results_address + "q4_03_near.jpg", near_image)
cv.imwrite(base_results_address + "q4_04_far.jpg", far_image)
cv.imwrite(base_results_address + "Q4_11_highpass.jpg", near_frequency_result)
cv.imwrite(base_results_address + "Q4_12_lowpass.jpg", far_frequency_result)
cv.imwrite(base_results_address + "Q4-13_hybrid_frequency.jpg", frequency_result)
cv.imwrite(base_results_address + "Q4-14_hybrid_near.jpg", domain_result)
cv.imwrite(base_results_address + "Q4-15_hybrid_far.jpg", realized_domain_result)
