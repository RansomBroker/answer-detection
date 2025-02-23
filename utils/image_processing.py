import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Load an image from the given path.
    """
    return cv2.imread(image_path)

def increase_image_brightness(image, factor=1.2):
    """
    Increase the brightness of the image by a given factor.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    brightened_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return brightened_image

def detect_brightness_level(image):
    """
    Detect the brightness level of the image.
    This calculates the average brightness (mean intensity) in grayscale.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray_image)  # Average intensity (brightness)
    return mean_intensity

def adjust_brightness_to_match_template(img_brightness, template_brightness, image, template):
    """
    Adjust the brightness of the image to match the brightness of the template.
    If the image's brightness is lower than the template, increase the image brightness.
    """
    if img_brightness < template_brightness:
        # Calculate the factor to adjust brightness
        factor = (template_brightness / img_brightness) -0
        adjusted_image = increase_image_brightness(image, factor)
        return adjusted_image
    return image

def crop_with_margin(image, boxes, margin=10):
    """
    Crop the image based on the bounding boxes with margin to ensure all boxes are included.
    """
    x_min = min([box[0][0][0] for box in boxes]) - margin
    x_max = max([box[2][0][0] for box in boxes]) + margin
    y_min = min([box[0][0][1] for box in boxes]) - margin
    y_max = max([box[2][0][1] for box in boxes]) + margin

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, image.shape[1])
    y_max = min(y_max, image.shape[0])

    cropped_image_with_margin = image[y_min:y_max, x_min:x_max]
    return cropped_image_with_margin