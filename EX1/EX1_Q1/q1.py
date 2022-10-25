from utilities import *
import matplotlib.pyplot as plt

image = cv.imread(base_images_address + 'dark.jpg')
image = image.astype(np.int16)

result_1 = image * 4
result_2 = gamma_transformation(image, 0.25)
result_3 = log_transformation(image, 1.2)

b_pdf1, g_pdf1, r_pdf1 = get_histogram(result_1)
b_pdf2, g_pdf2, r_pdf2 = get_histogram(result_2)
b_pdf3, g_pdf3, r_pdf3 = get_histogram(result_3)

b_cdf1, g_cdf1, r_cdf1 = get_cdf(result_1)
b_cdf2, g_cdf2, r_cdf2 = get_cdf(result_2)
b_cdf3, g_cdf3, r_cdf3 = get_cdf(result_3)

fig, ax = plt.subplots(3, 4)

plot_cdf(ax, 0, b_cdf1, g_cdf1, r_cdf1)
plot_cdf(ax, 1, b_cdf2, g_cdf2, r_cdf2)
plot_cdf(ax, 2, b_cdf3, g_cdf3, r_cdf3)

plot_pdf(ax, 0, b_pdf1, g_pdf1, r_pdf1)
plot_pdf(ax, 1, b_pdf2, g_pdf2, r_pdf2)
plot_pdf(ax, 2, b_pdf3, g_pdf3, r_pdf3)

ax[0, 3].imshow(cv.cvtColor(result_1.astype(np.uint8), cv.COLOR_BGR2RGB))
ax[1, 3].imshow(cv.cvtColor(result_2.astype(np.uint8), cv.COLOR_BGR2RGB))
ax[2, 3].imshow(cv.cvtColor(result_3.astype(np.uint8), cv.COLOR_BGR2RGB))

cv.imwrite(base_results_address + 'res01.jpg', result_1)
plt.show()
