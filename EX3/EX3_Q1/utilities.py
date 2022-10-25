from math import sqrt, pow, atan2

points_file_name = 'points.txt'
num_of_points = int

Data = {'x': [],
        'y': []}


def load_data(points_file_data, number_of_points, data):
    global Data
    mode = input("Please Enter mode. \n  Polar/Cartesian ?!\n").lower()
    while mode != 'polar' and mode != 'cartesian':
        print('please enter correct mode :(')
        mode = input().lower()

    for i in range(number_of_points):

        line = points_file_data.readline().split()
        if line:
            x, y = line[0], line[1]
            x, y = float(x), float(y)
            if mode == 'polar':
                vector_size = sqrt(pow(x, 2) + pow(y, 2))
                vector_angle = atan2(y, x)
                data['x'].append(vector_size)
                data['y'].append(vector_angle)
            elif mode == 'cartesian':
                data['x'].append(x)
                data['y'].append(y)
            Data['x'].append(x)
            Data['y'].append(y)

    return data
