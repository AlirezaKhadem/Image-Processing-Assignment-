import cv2 as cv
import numpy as np
import dlib
from scipy.spatial import Delaunay

images_address_base = 'images/'
results_address_base = 'results/'


def get_facial_landmarks(face_image):
    gray = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    key_points = []
    for face in detector(gray):
        landmarks = predictor(gray, face)

        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            key_points.append([x, y])

        key_points.append([0, 0])
        key_points.append([0, face_image.shape[0] // 2])
        key_points.append([face_image.shape[1] - 1, face_image.shape[0] // 2])
        key_points.append([0, face_image.shape[0] - 1])
        key_points.append([face_image.shape[1] // 2, 0])
        key_points.append([face_image.shape[1] - 1, 0])
        key_points.append([face_image.shape[1] // 2, face_image.shape[0] - 1])
        key_points.append([face_image.shape[1] - 1, face_image.shape[0] - 1])

    return key_points


def get_triangulation(facial_landmarks):
    return Delaunay(facial_landmarks)


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


def draw_delaunay_triangulation(src_image, facial_landmarks, triangulation):
    for triangle in triangulation.simplices:

        src_image_shape = src_image.shape
        boundary_rec = (0, 0, src_image_shape[1], src_image_shape[0])

        pt1 = facial_landmarks[triangle[0]]
        pt2 = facial_landmarks[triangle[1]]
        pt3 = facial_landmarks[triangle[2]]

        if rect_contains(boundary_rec, pt1) and rect_contains(boundary_rec, pt2) and rect_contains(boundary_rec, pt3):
            cv.line(src_image, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 0), 1)
            cv.line(src_image, (pt2[0], pt2[1]), (pt3[0], pt3[1]), (0, 255, 0), 1)
            cv.line(src_image, (pt3[0], pt3[1]), (pt1[0], pt1[1]), (0, 255, 0), 1)


def morph(first_image, last_image, facial_landmarks1, facial_landmarks2, triangulation, Num_of_frames):
    for frame_number in range(50):
        result = np.zeros(first_image.shape)

        for triangle in triangulation.simplices:
            face1_pt1 = facial_landmarks1[triangle[0]]
            face1_pt2 = facial_landmarks1[triangle[1]]
            face1_pt3 = facial_landmarks1[triangle[2]]

            face1_triangle = np.array([face1_pt1, face1_pt2, face1_pt3], np.int32)
            face1_triangle_boundary = cv.boundingRect(face1_triangle)

            x, y, w, h = face1_triangle_boundary
            cropped_face1_triangle_boundary = first_image[y:y + h, x:x + w]
            cropped_face1_triangle_vertices = np.array([[face1_pt1[0] - x, face1_pt1[1] - y],
                                                        [face1_pt2[0] - x, face1_pt2[1] - y],
                                                        [face1_pt3[0] - x, face1_pt3[1] - y]], np.int32)

            face2_pt1 = facial_landmarks2[triangle[0]]
            face2_pt2 = facial_landmarks2[triangle[1]]
            face2_pt3 = facial_landmarks2[triangle[2]]

            face2_triangle = np.array([face2_pt1, face2_pt2, face2_pt3])
            face2_triangle_boundary = cv.boundingRect(face2_triangle)

            x, y, w, h = face2_triangle_boundary
            cropped_face2_triangle_boundary = last_image[y:y + h, x:x + w]
            cropped_face2_triangle_vertices = np.array([[face2_pt1[0] - x, face2_pt1[1] - y],
                                                        [face2_pt2[0] - x, face2_pt2[1] - y],
                                                        [face2_pt3[0] - x, face2_pt3[1] - y]], np.int32)

            middle_frame_triangle = np.array((frame_number / (Num_of_frames - 1)) * face2_triangle + (
                    1 - frame_number / (Num_of_frames - 1)) * face1_triangle, np.float32)
            middle_frame_triangle_boundary = cv.boundingRect(middle_frame_triangle)

            middle_frame_pt1 = middle_frame_triangle[0]
            middle_frame_pt2 = middle_frame_triangle[1]
            middle_frame_pt3 = middle_frame_triangle[2]

            x, y, w, h = middle_frame_triangle_boundary
            cropped_middle_frame_triangle_mask = np.zeros((h, w), np.uint8)
            cropped_middle_frame_triangle_vertices = np.array([[middle_frame_pt1[0] - x, middle_frame_pt1[1] - y],
                                                               [middle_frame_pt2[0] - x, middle_frame_pt2[1] - y],
                                                               [middle_frame_pt3[0] - x, middle_frame_pt3[1] - y]],
                                                              np.int32)

            cv.fillConvexPoly(cropped_middle_frame_triangle_mask,
                              cropped_middle_frame_triangle_vertices, (255, 255, 255))

            affine_transform_from_1_to_middle = cv.getAffineTransform(
                cropped_face1_triangle_vertices.astype(np.float32),
                cropped_middle_frame_triangle_vertices.astype(
                    np.float32))
            affine_transform_from_2_to_middle = cv.getAffineTransform(
                cropped_face2_triangle_vertices.astype(np.float32),
                cropped_middle_frame_triangle_vertices.astype(
                    np.float32))

            warped_from_1_to_middle = cv.warpAffine(cropped_face1_triangle_boundary,
                                                    affine_transform_from_1_to_middle, (w, h))
            warped_from_2_to_middle = cv.warpAffine(cropped_face2_triangle_boundary,
                                                    affine_transform_from_2_to_middle, (w, h))

            warped_from_1_to_middle = cv.bitwise_and(warped_from_1_to_middle,
                                                     warped_from_1_to_middle,
                                                     mask=cropped_middle_frame_triangle_mask)
            warped_from_2_to_middle = cv.bitwise_and(warped_from_2_to_middle,
                                                     warped_from_2_to_middle,
                                                     mask=cropped_middle_frame_triangle_mask)

            triangle_area = result[y:y + h, x:x + w]
            subscribe = cv.bitwise_and(triangle_area, triangle_area, mask=cropped_middle_frame_triangle_mask)

            middle_triangle = (frame_number / (Num_of_frames - 1)) * warped_from_2_to_middle + (
                    1 - frame_number / (Num_of_frames - 1)) * warped_from_1_to_middle
            cv.imwrite('middle_triangle.jpg', middle_triangle)
            cv.waitKey(0)

            triangle_area = cv.add(middle_triangle, triangle_area)

            x1, y1, z1 = np.where(subscribe != 0)
            triangle_area[x1, y1, z1] = triangle_area[x1, y1, z1] - triangle_area[x1, y1, z1] / 2

            result[y:y + h, x:x + w] = triangle_area

        cv.imwrite(results_address_base + str(frame_number) + '.jpg', result)
