import numpy as np
import math
import time
import random

def get_plane_equation_from_points(para1, para2, para3):  
    x1, y1, z1 = para1[:3]
    x2, y2, z2 = para2[:3]
    x3, y3, z3 = para3[:3]
    a1 = x2 - x1 
    b1 = y2 - y1 
    c1 = z2 - z1 
    a2 = x3 - x1 
    b2 = y3 - y1 
    c2 = z3 - z1 
    a = b1 * c2 - b2 * c1 
    b = a2 * c1 - a1 * c2 
    c = a1 * b2 - b1 * a2 
    d = (- a * x1 - b * y1 - c * z1) 
    return a, b, c, d

def get_plane_equation_from_point_normal_vector(normal_vector, point):
    x,y,z = normal_vector[:3]
    A, B, C = point[:3]
    return x, y, z, -(x*A + y*B + z*C)

def get_distance_point_line_2d(point, line_equation):
    #ax + by + c = 0
    x1,y1 = point
    a,b,c = line_equation      
    distance = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    return distance

def get_line_equation_from_direction_vector_point(point, direction_vector):
    x,y = direction_vector
    A, B = point
    return y, -x, (-y*A + x*B )

# print(get_distance_point_line_2d([5,5],[0,1,-3]))

def get_angle_between_planes(para1, para2):
    vector_1 = para1[:3]
    vector_2 = para2[:3]
    # d = (a1 * a2) + (b1 * b2) + (c1 * c2) 
    # e1 = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1) 
    # e2 = math.sqrt( a2 * a2 + b2 * b2 + c2 * c2) 
    # d = d / (e1 * e2)
    # d = round(d,1) 
    # A = math.degrees(math.acos(d)) 

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = math.degrees(np.arccos(dot_product))
    return angle

def get_line_intersection_vector_from_two_planes(A, B):
    a,b,c,_ = A
    x,y,z,_ = B
    return [y*c - z*b, z*a - x*c, x*b - y*a]

def find_2d_corner(image, mask,object_depth,point_cloud,axis, start_loop, end_loop, direct):

    for i in range(start_loop, end_loop, direct):
        if axis == 'x':
            mask_collumn = mask[:,i]
        else:
            mask_collumn = mask[i]
        if mask_collumn.max() == 1:
            position_arr = np.where(mask_collumn == 1)
            temp_arr = []
            for temp in position_arr[0]:
                if axis == 'x':
                    if point_cloud[temp][i][0] == 0:
                        continue
                    try:
                        cur_dist = object_depth[temp][i]
                        if temp_dist > cur_dist:
                            temp_dist = cur_dist
                            box_corner = [i, temp]
                    except:
                        temp_dist = object_depth[temp][i]
                    
                    temp_arr.append(temp)
                else:
                    if point_cloud[i][temp][0] == 0:
                        continue
                    try:
                        cur_dist = object_depth[i][temp]
                        if temp_dist > cur_dist:
                            temp_dist = cur_dist
                            box_corner = [temp, i]
                    except:
                        temp_dist = object_depth[i][temp]

                    temp_arr.append(temp)

            if not temp_arr:
                continue
            try:
                box_corner
            except:
                continue
            break
    # image = cv.circle(image,tuple(box_corner), 5, (0,0,255), -1)
    return box_corner, image

def get_distance_point_plane(M, alpha):
    a1, b1, c1, d1 = alpha
    a2, b2, c2 = M
    num = (a1 * a2) + (b1 * b2) + (c1 * c2) + d1
    denom = math.sqrt( a1 * a1 + b1 * b1 + c1 * c1) 
    return num/denom

def get_distance_point_line_3d(M, direction_vector, P):
    # a1, b1, c1 = P
    # a2, b2, c2 = M
    x,y,z = direction_vector
    MP = M - P
    MPS = np.array([MP[1]*z - MP[2]*y, -MP[0]*x + MP[2]*z, MP[1]*x - MP[0]*y])
    num = np.sqrt(np.power(MPS[0], 2)+ np.power(MPS[1], 2)+ np.power(MPS[2], 2))
    nom = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
    return num/nom

def get_distance_two_point_3d(A, B):
    a1, b1, c1 = A
    a2, b2, c2 = B
    denom = math.sqrt( (a1 - a2)*(a1 - a2) + (b1 - b2)*(b1 - b2) + (c1 - c2)*(c1 - c2))
    return denom

def get_distance_two_point_2d(A, B):
    a1, b1 = A
    a2, b2 = B
    denom = math.sqrt((a1 - a2)*(a1 - a2) + (b1 - b2)*(b1 - b2))
    return denom

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))
def length(v):
  return math.sqrt(dotproduct(v, v))
def get_angle_two_vector(v1, v2):
  return math.degrees(math.acos(dotproduct(v1, v2) / (length(v1) * length(v2))))

def get_angle_two_line_3d(direction_vector_1, direction_vector_2):
    direction_vector_1 = np.array(direction_vector_1)
    direction_vector_2 = np.array(direction_vector_2)
    m1 = np.sqrt(np.power(direction_vector_1[0],2)+np.power(direction_vector_1[1],2)+np.power(direction_vector_1[2],2))
    m2 = np.sqrt(np.power(direction_vector_2[0],2)+np.power(direction_vector_2[1],2)+np.power(direction_vector_2[2],2))
    cosn = np.sum(direction_vector_1*direction_vector_2)/(m1*m2)
    return math.degrees(math.acos(cosn)) 

def Rx(theta):
  return np.matrix([[ 1, 0           , 0           ],
                   [ 0, math.cos(theta),-math.sin(theta)],
                   [ 0, math.sin(theta), math.cos(theta)]])
  
def Ry(theta):
  return np.matrix([[ math.cos(theta), 0, math.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-math.sin(theta), 0, math.cos(theta)]])
  
def Rz(theta):
  return np.matrix([[ math.cos(theta), -math.sin(theta), 0 ],
                   [ math.sin(theta), math.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def get_satellite_points_of_center_2d(center_point, normal_vector_2d, distance_to_center_point):
    line_equation = (normal_vector_2d[0], normal_vector_2d[1], -(normal_vector_2d[0]*center_point[0] + normal_vector_2d[1]*center_point[1]))
    x_value = (-line_equation[1]/line_equation[0], -line_equation[2]/line_equation[0])
    raw_value = np.power(x_value[1] - center_point[0],2) + np.power(center_point[1],2) - np.power(distance_to_center_point,2)
    level_1_value = 2*(x_value[0])*(x_value[1]-center_point[0]) - 2*(center_point[1])
    level_2_value = np.power(x_value[0],2) + 1

    # print("Handle Quadratic Equation: ax^2 + bx + c = 0")
    a = level_2_value
    b = level_1_value
    c = raw_value

    if a == 0:
        if b == 0:
            if c == 0:
                print("Countless!")
            else:
                print("impossible equation!")
        else:
            if c == 0:
                print("x = 0")
            else:
                print("x = ", -c / b)
    else:
        delta = np.power(b,2) - 4 * a * c
        if delta < 0:
            print("impossible equation!!")
        elif delta == 0:
            print("x = ", -b / (2 * a))
        else:
            # print("2 values!")
            y1 = int((-b - np.sqrt(delta)) / (2 * a))
            y2 = int((-b + np.sqrt(delta)) / (2 * a))
            x1 = int(x_value[0] * y1 + x_value[1])
            x2 = int(x_value[0] * y2 + x_value[1])
            return [x1,y1],[x2,y2]

def get_mask_boundary(mask_polygon):
    start_time = time.time()
    left_box_corner = right_box_corner = mask_polygon[0][0]
    top_box_corner = button_box_corner = mask_polygon[0][1]
    for coor in mask_polygon:
        if left_box_corner > coor[0]:
            left_box_corner = coor[0]
        if right_box_corner < coor[0]:
            right_box_corner = coor[0]
        if top_box_corner > coor[1]:
            top_box_corner = coor[1]
        if button_box_corner < coor[1]:
            button_box_corner = coor[1]
    end_time = time.time()
    # print('mask_polygon ellapsed time =', end_time - start_time)
    return left_box_corner, right_box_corner, top_box_corner, button_box_corner

def get_ramdom_rgb():
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    return (r,g,b)

def get_world_position(point_pos, cam_pos):
    # point_pos = error_compensatiton(point_pos)
    point_world_pos = [cam_pos[0]-point_pos[1],cam_pos[1]-point_pos[0],cam_pos[2]-point_pos[2]]
    return point_world_pos

def np_where_test(pc, mask):
    pc = np.reshape(pc, (-1,3))
    mask = np.reshape(mask,(-1,1))
    pc = np.where(mask == 1, pc, pc*0)

    pc = np.delete(pc, np.where(pc == np.array([0,0,0])), axis = 0)
    return pc

def get_pallet_surface(pc, plane_data):
    plane_equation = get_plane_equation_from_point_normal_vector(plane_data.vector, plane_data.point)
    # pc = np.reshape(pc, (-1,3))
    sr_pc = []
    for index in range(0,len(pc),5):
        point = pc[index]
        dist = get_distance_point_plane(point,plane_equation)
        if dist < 0.001:
            sr_pc.append(point)
    # sr_pc = np.delete(sr_pc, np.where(sr_pc == np.array([0,0,0])), axis = 0)
    return sr_pc

def get_x_axis(corner_point, cicle_points):
    direction_point,remaining_point = cicle_points
    direction_vector = np.array(direction_point) - np.array(corner_point)
    # (x−x1)(y2−y1)−(y−y1)(x2−x1)
    d = (remaining_point[0]- corner_point[0])* direction_vector[1] - (remaining_point[1]- corner_point[1])* direction_vector[0]
    if d > 0:
        return direction_point
    else:
        return remaining_point

def get_image_crop_range(click_pos, frame_shape):
    adjust_range = 0.1
    h,w = frame_shape
    sh = click_pos[1] - adjust_range
    eh = click_pos[1] + adjust_range
    sw = click_pos[0] - adjust_range*w/h
    ew = click_pos[0] + adjust_range*w/h

    if sh < 0:
        sh = 0
        eh = adjust_range*2
    elif eh> 1:
        eh = 1
        sh = 1 - adjust_range*2
    elif sw < 0:
        sw = 0
        ew = adjust_range*2*w/h
    elif ew > 1:
        ew = 1
        sw = 1 - adjust_range*2*w/h
    return [sh,eh,sw,ew]

def get_remaining_corner_parallelogram(corner_point, point_1, point_2):
    parallelogram_centroid = np.mean([point_1, point_2], axis=0)
    return parallelogram_centroid*2 - np.array(corner_point)