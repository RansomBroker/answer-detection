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
9. protobuf
10. python3.11.7 or newer

##### Instalasi package dan menjalankan program

1. Buat Environment baru dengan command `conda create --name answer-detection python=3.11.7` lalu aktivasi Environment dengan perintah `conda activate nama-env`
2. Install packages dengan perintah `pip install -r requirements.txt`
3. Unduh model ocr ``https://drive.google.com/file/d/1VCwunVXgQlzAoEhfJtVPJZ_zm-CfjDB6/view?usp=sharing`
4. Ekstrak dan taruh folder model ocr ke dalam folder model. Adapun struktur folder model seharusnya seperti dibawah ini:

   - model
     -- fine-tuning-small-handwriting
     -- best.pt

5. jalankan web server flask dengan menjalankan perintah `python app.py`
6. adapaun project ini terbagi menjadi 2 endpoint
   1. `/api/upload/omr` untuk omr
   2. `/api/upload/ocr` untuk ocr

`curl --location 'http://127.0.0.1:5000/api/upload' \
--form 'image=@"/C:/Users/yadis/Pictures/lembar valid1.jpg"'`
