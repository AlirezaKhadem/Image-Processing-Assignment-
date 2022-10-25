from utilities import *

indexes = [[-1, -1], [-1, 0], [-1, 1],
           [0, -1], [0, 0], [0, 1],
           [1, -1], [1, 0], [1, 1]]

image = cv.imread(base_images_address + 'tasbih.jpg')

image = cv.medianBlur(image, 5)
image = cv.bilateralFilter(image, -1, 50, 12)
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

edge_detected_image_ = get_gradient(gray_image)


def mouse_handler(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        vertices_.append([x, y])
        cv.circle(image, (x, y), 2, (0, 0, 255), 2)
    if event == cv.EVENT_RBUTTONDOWN:
        centers.append([x, y])
        cv.circle(image, (x, y), 2, (255, 0, 0), 5)


cv.namedWindow('window', cv.WINDOW_NORMAL)
cv.setMouseCallback('window', mouse_handler)

while True:
    cv.imshow('window', image)

    key = cv.waitKey(100)
    if key == 27:
        cv.destroyAllWindows()
        break

iteration(edge_detected_image_, vertices_, indexes, centers, 100, 48500, 25000, k_window=3, max_iteration=200)
