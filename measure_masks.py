import os
import cv2
import csv
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

"""
    This script is for measuring the ostracods from given mask images.
    1. Load mask image, doing measurament
    2. Draw the measurement result on the original image
    3. Save the image with measurement result
"""

def build_img_folder_index(img_folder):
    """
        Build the index of the image folder
        The index is a dict with key as the image name and value as the image path
    """
    img_index = {}
    for root, dirs, files in os.walk(img_folder):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
                img_index[file] = os.path.join(root, file)
    return img_index

def get_line_equation(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def find_perpendicular_line(m1, x, y):
    # Calculate the negative reciprocal of the slope of the given line
    m2 = -1 / m1

    # Calculate the y-intercept of the perpendicular line
    b2 = y - m2 * x

    # Combine m2 and b2 to form the equation of the perpendicular line
    return m2, b2

def calculate_distance(a, b1, b2):
    distance = abs(b2 - b1) / np.sqrt(1 + a**2)
    return distance

def get_height_measurements(coordinates, width_indices):
    """
    Get the height measurements from the coordinates and mask
    """ 
    wp_1, wp_2 = coordinates[width_indices[0]][0], coordinates[width_indices[1]][0]
    # create resoultion number of points on the width line evenly
    m,b = get_line_equation(wp_1[0], wp_1[1], wp_2[0], wp_2[1])
    max_b = b
    min_b = b
    max_idx = 0
    min_idx = 1
    for points in range(len(coordinates)):
        x_p, y_p = coordinates[points][0]
        b_p = y_p - m * x_p
        if b_p > max_b:
            max_b = b_p
            max_idx = points
        if b_p < min_b:
            min_b = b_p
            min_idx = points
    # define height a the distance between the max_b line and min_b line
    height = calculate_distance(m, max_b, min_b)
    return height, (max_idx, min_idx)

def draw_measurements_sp(image, coordinates, width_indices, height_indices, scalebar_length):
    wp_1, wp_2 = coordinates[width_indices[0]][0], coordinates[width_indices[1]][0]
    hp_1, hp_2 = coordinates[height_indices[0]][0], coordinates[height_indices[1]][0]
    # check if the points are NaN
    if np.isnan(wp_1[0]) or np.isnan(wp_1[1]) or np.isnan(wp_2[0]) or np.isnan(wp_2[1]) or np.isnan(hp_1[0]) or np.isnan(hp_1[1]) or np.isnan(hp_2[0]) or np.isnan(hp_2[1]):
        return image
    # copy the image data to a new cv2 image
    np_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Create a cv2 image from the NumPy array
    cv2_image = cv2.cvtColor(np_array, cv2.COLOR_BGR2RGB)
    # Fill Circle on the height and width points
    cv2.circle(cv2_image, (int(wp_1[0]), int(wp_1[1])), 5, (255, 0, 0), -1)
    cv2.circle(cv2_image, (int(wp_2[0]), int(wp_2[1])), 5, (255, 0, 0), -1)
    cv2.circle(cv2_image, (int(hp_1[0]), int(hp_1[1])), 5, (255, 0, 0), -1)
    cv2.circle(cv2_image, (int(hp_2[0]), int(hp_2[1])), 5, (255, 0, 0), -1)
    # draw line of the width points
    cv2.line(cv2_image, (int(wp_1[0]), int(wp_1[1])), (int(wp_2[0]), int(wp_2[1])), (255, 0, 0), 2)
    mid_point = ((wp_1[0]+wp_2[0])//2, (wp_1[1]+wp_2[1])//2)
    m,b = get_line_equation(wp_1[0], wp_1[1], wp_2[0], wp_2[1])
    b_1 = hp_1[1] - m * hp_1[0]
    b_2 = hp_2[1] - m * hp_2[0]
    m_p, b_p = find_perpendicular_line(m, mid_point[0], mid_point[1])
    x_1_p = (b_p - b_1) / (m - m_p)
    y_1_p = m_p * x_1_p + b_p
    x_2_p = (b_p - b_2) / (m - m_p)
    y_2_p = m_p * x_2_p + b_p
    if np.isnan(x_1_p) or np.isnan(y_1_p) or np.isnan(x_2_p) or np.isnan(y_2_p):
        return image
    # draw line between x_1_p, y_1_p and x_2_p, y_2_p
    cv2.line(cv2_image, (int(x_1_p), int(y_1_p)), (int(x_2_p), int(y_2_p)), (255, 0, 0), 2)
    # draw dash line between hp_1 and x_1_p, y_1_p
    lineType = cv2.LINE_AA
    pattern = [10, 5]
    cv2.line(cv2_image, (int(hp_1[0]), int(hp_1[1])), (int(x_1_p), int(y_1_p)), color=(125, 125, 0), thickness=1, lineType=lineType)
    # draw dash line between hp_2 and x_2_p, y_2_p
    cv2.line(cv2_image, (int(hp_2[0]), int(hp_2[1])), (int(x_2_p), int(y_2_p)), color=(125,125, 0), thickness=1, lineType=lineType)
    # draw the length and width text on the bottom left corner of the image, font size 12, unit: μm, with white background
    # using cv2, HEIGHT is the distance between x_1_p, y_1_p and x_2_p, y_2_p
    width = np.sqrt((wp_1[0]-wp_2[0])**2+(wp_1[1]-wp_2[1])**2)/scalebar_length
    height = np.sqrt((x_1_p-x_2_p)**2+(y_1_p-y_2_p)**2)/scalebar_length
    image_height, image_width, _ = cv2_image.shape
    cv2.rectangle(cv2_image, (0, image_height), (200, image_height-60), (255, 255, 255), -1)
    cv2.putText(cv2_image, f'height: {format(height, ".2f")[:4]}um', (10, image_height -10), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(cv2_image, f'length: {format(width, ".2f")[:4]}um', (10, image_height-40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)
    # Convert the cv2 image back to a PIL image
    modified_image = Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    #modified_image = Image.fromarray(cv2_image)
    return modified_image, width, height

def mask_measurament(mask, image, image_name, scalebar_length):
    """
        Do the measurament on the mask image.
        Define the length as the longest distance between two points on the mask edge.
        Define width as the longest distance on the direction perpendicular to the length direction.
        Input:
            mask: the mask image
            image_name: the name of image, use to check the original image from index
        Output:
            original_image with drawn measurament result
            list contains measurament result
    """
    ostracod_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    base_ostracod_contour = ostracod_contours[0]
    for ostracod_contour in ostracod_contours[1:]:
        base_ostracod_contour = np.concatenate((base_ostracod_contour, ostracod_contour), axis=0)
    #shepe = ostracod_contours[0].shape
    distance_matrix = np.sqrt(((base_ostracod_contour[:, None, :] - base_ostracod_contour) ** 2).sum(-1))
    width_indices = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    _, height_indices = get_height_measurements(base_ostracod_contour, width_indices)
    image, length, height = draw_measurements_sp(image, base_ostracod_contour, width_indices, height_indices,scalebar_length)
    return image, [image_name, length, height]

def main():
    # define the folder of mask images
    mask_folder = "D:\\working_dir\\measurament\\wagner_masks_cleaned"
    # define the folder of original images
    image_folder = "C:\\Users\\admin\\Desktop\\Wagner copy"
    # define the folder to save the images with measurement result
    save_folder = "D:\\working_dir\\measurament\\result"
    # get the index of the mask folder
    mask_index = build_img_folder_index(mask_folder)
    # get the index of the image folder
    image_index = build_img_folder_index(image_folder)
    # check if the image and mask index are the same
    # if mask_index.keys() != image_index.keys():
    #     print("The image and mask index are not the same")
    #     return
    measurements = [['image','length','height']]
    measurement_save_path = os.path.join(save_folder, 'measurement.csv')
    scalebar_lengthes = pd.read_csv('D:\\working_dir\\measurament\\scalebar_length.csv')
    # loop through the mask index
    for image_name, mask_path in tqdm(mask_index.items()):
        # get the original image path
        image_path = image_index[image_name]
        # load the mask image
        mask = cv2.imread(mask_path, 0)
        # load the original image
        image = cv2.imread(image_path)
        # get the scalebar length of the image
        scalebar_length = scalebar_lengthes[scalebar_lengthes['image'] == image_name]['scalebar_length'].values[0]
        # do the measurament on the mask image
        image, measurement = mask_measurament(mask, image, image_name,scalebar_length)
        # save the image with measurement result
        image.save(os.path.join(save_folder, image_name))
        # print the measurement result
        measurements.append(measurement)
    # save the measurement result
    with open(measurement_save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(measurements)

if __name__ == '__main__':
    main()