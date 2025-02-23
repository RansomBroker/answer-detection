import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import math

def align_images(image, template, maxFeatures=5000, keepPercent=0.1, debug=False):
    # convert both the input image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(maxFeatures)
    
    # Find keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(templateGray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(imageGray, None)

     # Ensure descriptors are not None and of the same type
    if descriptors1 is None or descriptors2 is None:
        raise ValueError("Descriptors cannot be None")
    
    if descriptors1.dtype != descriptors2.dtype:
        descriptors1 = descriptors1.astype(descriptors2.dtype)
    
    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Descriptors must have the same number of columns")
    
    
    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Use the top matches for homography
    num_good_matches = int(len(matches) * keepPercent)  # Take the best 10% matches
    good_matches = matches[:num_good_matches]
    
    # Extract location of good matches
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography matrix
    H, mask = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)
    
    # Warp image using the homography matrix
    aligned_image = cv2.warpPerspective(imageGray, H, (templateGray.shape[1], templateGray.shape[0]))

    return aligned_image

def detect_filled_rectangles_with_adjusted_filters(image):
    """
    Detect rectangles that are filled with dark color (near black), with adjusted filters:
    - Minimum area (lower threshold)
    - Correct aspect ratio (relaxed range).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filled_rectangles = []

    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 300:
                cropped = gray[y:y+h, x:x+w]
                mean_intensity = np.mean(cropped)
                if mean_intensity < 100:
                    aspect_ratio = float(w) / h
                    if 0.6 < aspect_ratio < 1.5:
                        filled_rectangles.append(approx)

    return filled_rectangles

def detect_circles_in_cropped_image(image):
    """
    Detect circles within the cropped image that are filled with dark color.
    """
    # Convert to grayscale for Hough Circle detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold using Otsu's method to the grayscale image
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    blurred = cv2.GaussianBlur(binary_image, (11, 11), 0)
    
    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT, dp=1.2, minDist=5, param1=50, param2=24, minRadius=5, maxRadius=15
    )

    filled_circles = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Crop the area around the circle
            cropped = binary_image[y-r:y+r, x-r:x+r]
            
            # Calculate the mean intensity in the cropped area
            mean_intensity = np.mean(cropped)
            if mean_intensity > 150:  # White intensity check for filled circles
                filled_circles.append((x, y, r))

    return filled_circles

def find_matching_answer(answerJsonPath, detected_circle, threshold=10):
    answer_selected = ["-" for i in range(60)]
    user_id_list = [0,0,0,0,0,0,0,0,0]

    with open(answerJsonPath, 'r') as file:
        answer_position = json.load(file)
    
    # Loop through each answer position
    for answer_idx, answer_list in enumerate(answer_position):
        for idx, (cx, cy) in enumerate(answer_list):
            for (x,y,r) in detected_circle:
                # Calculate distance
                distance = math.sqrt((x - cx)**2 + (y - cy)**2)
                if distance <= threshold:
                    if answer_idx < 9 :
                        user_id_list[idx] = answer_idx 
                    if answer_idx > 9:
                        answer_selected[answer_idx-10] = chr(idx + 65)

    # Formated output
    user_id = "".join(map(str, user_id_list))

    return user_id, answer_selected
                    



