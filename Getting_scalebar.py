import os
import cv2
import numpy as np
import csv

"""
Getting the length of the scale bar from the images and summarize into a table
"""


def get_scalebar_length(image, scalebar_length):
    """
    :param image: the image with scale bar
    :return: the length of the scale bar in um
    The scale bar is in white color, so setting aggressive threshold to get the scale bar.
    Using morphological closing to connect the broken part of the scale bar to increase quality of the extraction.
    As the countour of the image contains some noise, so the largest contour is selected assumed to be the scale bar.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # get sub image of the bottom right corner from point (960, 160)
    gray = gray[1040:, 600:]
    _, thresholded_image = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    # As the scale bar in horizontal, the kernel is set to focus on horizontal connection
    kernel = np.ones((6,1),np.uint8)
    morph = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
    rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
    w = rightmost[0] - leftmost[0]
    return w/scalebar_length

if __name__ == '__main__':
    # assumption: the length of scare bar is 200 um
    image_folder = 'c:/users/admin/desktop/wagner copy'
    save_file = 'D:\\working_dir\\measurament\\scalebar_length.csv'
    scalebar_length = 200
    results = [['image', 'scalebar_length']]
    for item in os.listdir(image_folder):
        sub_folder = os.path.join(image_folder, item)
        if os.path.isdir(sub_folder):
            print('Processing folder: ', sub_folder)
            for img in os.listdir(sub_folder):
                if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg'):
                    print('Extracting image: ', img)
                    image = cv2.imread(os.path.join(sub_folder, img))
                    img_path = os.path.join(sub_folder, img)
                    img_data = cv2.imread(img_path)
                    img_pixel = img_data.shape[0] * img_data.shape[1]
                    if img_pixel > 3840000:
                        print(f'Image size of {img} is too large, skip.')
                        continue
                    image_scale = get_scalebar_length(image, scalebar_length)
                    results.append([img, image_scale])
    with open(save_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results)
