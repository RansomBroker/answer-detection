import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_filled_rectangles(image, rectangles):
    """
    Draw green rectangles around the detected filled rectangles.
    """
    img_with_rectangles = image.copy()
    for rect in rectangles:
        cv2.drawContours(img_with_rectangles, [rect], -1, (0, 255, 0), 2)

    return img_with_rectangles

def display_image(image):
    """
    Display the image with matplotlib.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
