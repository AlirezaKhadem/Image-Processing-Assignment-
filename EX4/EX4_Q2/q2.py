from utilities import *

image_face1 = cv.imread(images_address_base + 'BradPitt.jpg')
image_face2 = cv.imread(images_address_base + 'DiCaprio.jpg')

facial_landmarks1 = get_facial_landmarks(image_face1)
facial_landmarks2 = get_facial_landmarks(image_face2)

triangulation = get_triangulation(facial_landmarks1)

morph(image_face1, image_face2, facial_landmarks1, facial_landmarks2, triangulation, 50)
