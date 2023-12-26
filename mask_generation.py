import numpy as np
import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

'''
The script is generating the masks for the target images then save to png files.
'''


def anti_bkg_test(mask):
    """
    Take binary mask image as input, test if the mask is in background.
    Define background as if the mask inclide any aera of 4 corners of the image. We take 10*10 windows at the 4 corners. Test if the mask sum in those corners are all 0.
    mask: binary mask image
    return: True if the mask is in background, False if not.
    """
    # test the top left corner
    if not np.sum(mask[:10, :10]) == 0:
        return True
    # test the top right corner
    if not np.sum(mask[:10, -10:]) == 0:
        return True
    # test the bottom left corner
    if not np.sum(mask[-10:, :10]) == 0:
        return True
    # test the bottom right corner
    if not np.sum(mask[-10:, -10:]) == 0:
        return True
    return False

def mask2img(anns):
    """
    Transfer the largest mask into image.
    anns: annotation returned by the segmentation anything annotator
    """
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    sel_mask = sorted_anns[0]['segmentation']
    for ann in sorted_anns:
        centroid_pt = [ann['segmentation'].shape[0] // 2, ann['segmentation'].shape[1] // 2]
        # calculte the sum of the mask in centroid +/- 10 pixels
        #sum_mask = np.sum(ann['segmentation'][centroid_pt[0] - 10:centroid_pt[0] + 10, centroid_pt[1] - 10:centroid_pt[1] + 10])
        # test by if the sum of the mask is larger than 400 (close to 21*21)
        #if ann['segmentation'][centroid_pt[0]][centroid_pt[1]] ==1 :#sum_mask >= 400:
        if not anti_bkg_test(ann['segmentation']):
            sel_mask = ann['segmentation']
            break
    #sel_mask = sorted_anns[0]['segmentation']
    mask_img = (sel_mask * 255).astype(np.uint8)
    return mask_img

def select_best_mask(masks, scores):
    """
    Select the mask with the highest score.
    masks: list of masks returned by the segmentation anything predictor
    scores: list of scores returned by the segmentation anything predictor
    """
    best_mask = masks[0]
    best_score = scores[0]
    for mask, score in zip(masks, scores):
        if score > best_score:
            best_mask = mask
            best_score = score
    mask_img = (best_mask * 255).astype(np.uint8)
    return mask_img


def sam_annotation(img, mask_generator):
    """
    Generate the annotation for the image.
    img: image in numpy array
    sam: segmentation anything model
    """
    anns = mask_generator.generate(img)
    mask_img = mask2img(anns)
    return mask_img

def predictor_annotation(img, mask_predictor):
    """
    Generate the annotation for the image. Assuming the object is at the centroid of the image to avoid accept background when using the largest mask.
    img: image in numpy array
    mask_predictor: segmentation anything predictor
    """
    # coordinates of four corners of the image
    coords = np.array([[0, 0], [0, img.shape[1]], [img.shape[0], 0], [img.shape[0], img.shape[1]]])
    centroid_pt = np.array([[800, 600], [0, 0], [0, img.shape[1]], [img.shape[0], 0], [img.shape[0], img.shape[1]]])
    centroid_label = np.array([1,0,0,0,0])
    mask_predictor.set_image(img)
    masks, scores, _ = mask_predictor.predict(point_coords=centroid_pt, point_labels=centroid_label, multimask_output=True)
    mask_img = select_best_mask(masks, scores)
    return mask_img

if __name__ == '__main__':
    sam_checkpoint = "../segment-anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:1"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(model=sam,
    points_per_side=3,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.96,
    crop_n_layers=2,
    crop_n_points_downscale_factor=1,
    min_mask_region_area=1000,  # Requires open-cv to run post-processing
    )
    mask_predictor = SamPredictor(sam)
    image_folder = "D:\\working_dir\\segment-anything\\notebooks\\sugery_desk" #'/mnt/c/users/admin/desktop/wagner copy'
    save_folder = 'wagner_masks_fix'
    for item in os.listdir(image_folder):
        sub_folder = os.path.join(image_folder, item)
        if os.path.isdir(sub_folder):
            print('Processing folder: ', sub_folder)
            for img in os.listdir(sub_folder):
                if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.jpeg'):
                    print('Annotating image: ', img)
                    img_path = os.path.join(sub_folder, img)
                    img_data = cv2.imread(img_path)
                    img_pixel = img_data.shape[0] * img_data.shape[1]
                    if img_pixel > 7680000:
                        print(f'Image size of {img} is too large, skip.')
                        continue
                    #img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
                    mask_img = predictor_annotation(img_data, mask_predictor)
                    mask_img = sam_annotation(img_data, mask_generator)
                    save_path = os.path.join(save_folder, img)
                    cv2.imwrite(save_path, mask_img)



