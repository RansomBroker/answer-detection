from utils import *
import cv2
from ultralytics import YOLO
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
import time


# OMR
def scan_answer(template_image, image_path, answer_json):
    """
    Full processing pipeline:
    - Detect rectangles.
    - Increase image brightness.
    - Crop image based on bounding sboxes.
    - Detect circles in the cropped image.
    """
    img = load_image(image_path)
    template = load_image(template_image)

    # Detect filled rectangles
    filled_rectangles = detect_filled_rectangles_with_adjusted_filters(img)
    template_rectangles = detect_filled_rectangles_with_adjusted_filters(template)

    if len(filled_rectangles) < 27 :
        return {"error": "No filled rectangles detected.", "code": 1, "answer_selected": None, "user_id_detected": None}

    # Draw rectangles on the image
    img_with_rectangles = draw_filled_rectangles(img, filled_rectangles)
    template_with_rectangles = draw_filled_rectangles(template, template_rectangles)

    # Detect brightness of both images
    img_brightness_level = detect_brightness_level(img_with_rectangles)
    template_brightness_level = detect_brightness_level(template_with_rectangles)

    print(f"Image Brightness Level: {img_brightness_level}")
    print(f"Template Brightness Level: {template_brightness_level}")

    if img_brightness_level < 125:
        return {"error": "Image brightness is too low.", "code": 2, "answer_selected": None, "user_id_detected": None}

    # Crop the image with detected bounding boxes
    cropped_image_with_margin = crop_with_margin(img_with_rectangles, filled_rectangles)
    template_cropped_image_with_margin = crop_with_margin(template_with_rectangles, template_rectangles)

    # Display images
    # display_image(img_with_rectangles)  # Show image with detected rectangles
    # display_image(cropped_image_with_margin)  # Show cropped image with detected circles
    # display_image(template_with_rectangles)  # Show template image with rectangles
    # display_image(template_cropped_image_with_margin)  # Show cropped template image

    # Align Image
    aligned_image = align_images(cropped_image_with_margin, template_cropped_image_with_margin, debug=False)

    # Ensure the aligned image is in BGR format
    if len(aligned_image.shape) == 2 or aligned_image.shape[2] == 1:
        aligned_image = cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2BGR)

    # Detect dark circles on the aligned image
    lower_image_brightness = increase_image_brightness(aligned_image, 0.5)
    print(detect_brightness_level(lower_image_brightness))
    dark_circles_in_aligned_image = detect_circles_in_cropped_image(lower_image_brightness)

    # If dark circles are detected, draw them on the aligned image in red
    for (x, y, r) in dark_circles_in_aligned_image:
        cv2.circle(aligned_image, (x, y), r, (0, 0, 255), 4)  

    # Display aligned image with dark circles
    # display_image(aligned_image)  

    # Matched Answer
    (user_id, answer_selected) = find_matching_answer(answer_json, dark_circles_in_aligned_image)

    return {"error": None, "code": 0, "answer_selected": answer_selected, "user_id": user_id}

# OCR
def detect_and_ocr(image_path):
    try:
        # Load YOLO
        model = YOLO("model/best.pt")

        # Load TrOCR
        processor_path = "microsoft/trocr-small-handwritten"
        model_trocr_path = "model/fine-tuning-small-handwriting"

        processor = TrOCRProcessor.from_pretrained(processor_path)
        model_trocr = VisionEncoderDecoderModel.from_pretrained(model_trocr_path, local_files_only=False)

        # Load image 
        image = cv2.imread(image_path)

        # Detect objects
        results = model([image])

        # Cropped image
        cropped_images = []

        for result in results:
            for box in result.boxes:
                # Crop image
                cropped_images.append(image[int(box.xyxy[0][1]):int(box.xyxy[0][3]), int(box.xyxy[0][0]):int(box.xyxy[0][2])])

        # Do OCR
        batch_size = 6  # Adjust the batch size according to your GPU memory
        all_texts = []
        inference_times = []

        for i in range(0, len(cropped_images), batch_size):
            batch_images = cropped_images[i:i + batch_size]
            pixel_values = processor(batch_images, return_tensors="pt").pixel_values

            start_time = time.time()
            generated_ids = model_trocr.generate(pixel_values)
            end_time = time.time()

            inference_time = end_time - start_time
            inference_times.append(inference_time)

            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_texts.extend(generated_texts)
            print(inference_time)

        # Sort the texts from small to large
        all_texts.sort(key=len)

        return {"answers": all_texts, "error": None}

    except Exception as e:
        return {"answers": [], "error": str(e)}


# Run the full processing pipeline on the image with brightness adjustment
# image_path = 'images\lembar_1.jpg'  # Adjust the path as necessary
# print(scan_answer("images\lembar jawaban.jpg", image_path, answer_json="answer_position.json"))

# Example usage
# result = detect_and_ocr('images/lembar jawaban manual.jpg')
# print(result)
