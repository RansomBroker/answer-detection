import cv2
import json
from utils import *

# Variabel untuk menyimpan koordinat klik
coordinates = []
current_coordinates = []

# Fungsi untuk menangani event klik mouse
def click_event(event, x, y, flags, params): 
    global current_coordinates
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDOWN: 
        # displaying the coordinates on the Shell 
        print(x, ' ', y) 
        # displaying the coordinates on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(img, str(x) + ',' + str(y), (x, y), font, 0.25, (255, 0, 0), 2) 
        cv2.imshow('image', img) 
  
        # Menyimpan koordinat ke dalam array saat klik kiri
        current_coordinates.append([x, y])
        
    # checking for right mouse clicks      
    if event == cv2.EVENT_RBUTTONDOWN: 
        # displaying the coordinates on the Shell 
        print(x, ' ', y) 
        # displaying the coordinates on the image window 
        font = cv2.FONT_HERSHEY_SIMPLEX 
        b = img[y, x, 0] 
        g = img[y, x, 1] 
        r = img[y, x, 2] 
        cv2.putText(img, str(b) + ',' + str(g) + ',' + str(r), (x, y), font, 0.25, (255, 255, 0), 2) 
        cv2.imshow('image', img)

# Fungsi untuk mendeteksi saat tombol Enter ditekan dan menyimpan hasil
def process_image_with_brightness_and_circles(template_image):
    # Menampilkan gambar yang telah diproses
    global img, current_coordinates, coordinates
    
    template = load_image(template_image)

    template_rectangles = detect_filled_rectangles_with_adjusted_filters(template)

    template_with_rectangles = draw_filled_rectangles(template, template_rectangles)

    template_cropped_image_with_margin = crop_with_margin(template_with_rectangles, template_rectangles)
    
    img = template_cropped_image_with_margin
    cv2.imshow('image', img)

    # Mengaitkan fungsi click_event dengan jendela gambar
    cv2.setMouseCallback('image', click_event)

    # Menunggu input pengguna sebelum menutup jendela
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Tombol Enter (ASCII 13)
            # Simpan current coordinates ke coordinates
            if current_coordinates:  # Hanya simpan jika current_coordinates tidak kosong
                coordinates.append(current_coordinates)
                # Reset current coordinates
                current_coordinates = []
                print("Data saved:", coordinates)  # Menampilkan data koordinat yang sudah disimpan
        elif key == 27:  # Tombol ESC (ASCII 27)
            # Menyimpan data koordinat ke dalam file JSON saat tombol ESC ditekan
            with open("answer_position1.json", "w") as json_file:
                json.dump(coordinates, json_file, indent=4)
            print("Data saved to answer_position.json")
            break

    cv2.destroyAllWindows()

# Run the full processing pipeline on the image with brightness adjustment
process_image_with_brightness_and_circles("images\lembar jawaban.jpg")