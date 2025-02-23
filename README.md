## Petunjuk penggunaan OMR

#### Requirenment

1. opencv 4.0
2. flask
3. pytorch
4. transformer
5. ultralystic
6. sentecepiece
7. matplotlib
8. imutils
9. python3.11.7 or newer

##### Instalasi package dan menjalankan program

1. Instal requirenment.txt dengan perintah `conda create --name answer-detection --file requirements.txt`
2. Unduh model ocr ``https://drive.google.com/file/d/1VCwunVXgQlzAoEhfJtVPJZ_zm-CfjDB6/view?usp=sharing`
3. Ekstrak dan taruh folder model ocr ke dalam folder model. Adapun struktur folder model seharusnya seperti dibawah ini:

   - model
     -- fine-tuning-small-handwriting
     -- best.pt

4. jalankan web server flask dengan menjalankan perintah `python app.py`
5. adapaun project ini terbagi menjadi 2 endpoint
   1. `/api/upload/omr` untuk omr
   2. `/api/upload/ocr` untuk ocr

`curl --location 'http://127.0.0.1:5000/api/upload' \
--form 'image=@"/C:/Users/yadis/Pictures/lembar valid1.jpg"'`
