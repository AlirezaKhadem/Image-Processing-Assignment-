from utilities import *

apple_image = cv.imread(base_address_image + '090_warped.jpg')
orange_image = cv.imread(base_address_image + '270_warped.jpg')

apple_image_copy = apple_image.copy()
param = {'apple_image': apple_image_copy}
cv.namedWindow('click for vertices of Mask. press ESC to continue')
cv.setMouseCallback('click for vertices of Mask. press ESC to continue', mouse_handler, param)

while True:
    cv.imshow('click for vertices of Mask. press ESC to continue', apple_image_copy)

    key = cv.waitKey(10)
    if key == 27:
        mask = get_mask(apple_image.shape, points)

        result = pyramid_blend(apple_image.astype(np.float64), orange_image.astype(np.float64), mask[:, :, 0])

        cv.imwrite(base_address_results + 'res2.jpg', result)
        cv.destroyAllWindows()
        break
