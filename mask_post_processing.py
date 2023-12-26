import os
import cv2
import numpy as np
from tqdm import tqdm

"""
    This script is for pre-peocessing the masks of images.
    The masks are in the format of .png, and the background is black.
        1. Keep only the largest connected component.
        2. Fill the holes inside the mask.
"""

def keep_largest_component(mask_img):
    """
    For filtering the extra components in the mask.
    input:
        mask_img: the mask image in numpy array.
    output:
        mask_img: the mask image with only the largest component.
    """
    _, binary_mask = cv2.threshold(mask_img, 199, 255, cv2.THRESH_BINARY)
    # Find countours
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    # Create a mask image for drawing purposes
    mask_img = np.zeros_like(mask_img)
    # Draw the largest contour
    cv2.drawContours(mask_img, [largest_contour], 0, (255, 255, 255), -1)
    return mask_img

def fill_holes(mask_img):
    """
    For filling the holes in the mask.
    input:
        mask_img: the mask image in numpy array.
    """
    _, binary_mask = cv2.threshold(mask_img, 199, 255, cv2.THRESH_BINARY)

    # Define the structuring element for closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Perform closing operation to fill the holes
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask


if __name__ == '__main__':
    mask_dir = 'D:\working_dir\measurament\wagner_masks'
    save_dir = 'D:\working_dir\measurament\wagner_masks_cleaned'
    for mask_name in tqdm(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, mask_name)
        if not os.path.isfile(mask_path):
            continue
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_img = keep_largest_component(mask_img)
        mask_img = fill_holes(mask_img)
        save_path = os.path.join(save_dir, mask_name)
        cv2.imwrite(save_path, mask_img)

