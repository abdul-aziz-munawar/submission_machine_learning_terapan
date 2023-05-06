# Prediksi Harga Tiket Pesawat di India - Abdul Aziz Munawar

## Domain Proyek

**Latar Belakang**  
Pesawat merupakan salah satu moda transportasi yang dapat digunakan untuk bepergian, baik untuk urusan bisnis, liburan maupun urusan lainnya. Untuk dapat menggunakan jasa moda transportasi ini, seseorang harus memiliki tiket pesawat.

Harga tiket yang terlalu mahal akan membuat seseorang berpikir berkali-kali untuk menggunakan jasa moda transportasi pesawat. Sebaliknya, harga tiket pesawat yang terlalu murah, akan menyebabkan perusahaan maskapai penerbangan tidak dapat memaksimalkan keuntungan bagi perusahaan, bahkan jika tidak ditangani secara serius, biaya operasional tidak akan sebanding dengan keuntungan yang didapatkan, hal ini akan menyebabkan kerugian bagi perusahaan maskapai penerbangan.

**Oleh sebab itu, diperlukan suatu aplikasi yang dapat memprediksi secara akurat, harga tiket pesawat yang ideal untuk ditawarkan kepada pelanggan, sehingga harga tiket akan seimbang (tidak terlalu mahal, juga tidak terlalu murah).**

Pada kasus ini, aplikasi *machine learning* secara spesifik akan memprediksi harga ideal untuk tiket, sehingga dapat dijadikan acuan oleh maskapai penerbangan di India, dalam menetapkan keputusan harga tiket bagi calon pengguna jasa layanan pesawatnya.

**Alasan Penting Yang Mendasari Proyek Ini**:
- Alasan penting yang mendasari bahwa permasalahan harga tiket harus diselesaikan, yaitu sebagai berikut:
    - harga tiket yang tidak terlalu mahal dapat menyebabkan seseorang untuk berpikir beberapa kali untuk menggunakan jasa moda transportasi melalui pesawat yang disediakan perusahaan x.
    - harga tiket yang terlalu murah dapat menyebabkan keuntungan perusahaan x tidak optimal, bahkan dapat menyebabkan kerugian. Alasannya karena, biaya operasional tidak sebanding dengan keuntungan yang didapatkan.
    - untuk menyelesaikan permasalahan tersebut, maka akan dibuat aplikasi yang dapat memprediksi harga ideal untuk tiket pesawat penerbangan (dalam kasus ini, prediksi spesifik hanya akan menampilkan harga ideal pesawat maskapai penerbangan di India).
    - aplikasi ini akan memanfaatkan teknologi machine learning serta bahasa pemrograman Python dalam membuat prediksi harga ideal untuk menjadi bahan keputusan bagi maskapai penerbangan dalam menentukan harga tiket pesawat.
- Hasil riset terkait:
   - [Predicting Flight Prices in India](https://www.researchgate.net/profile/Tarun-Devireddy/publication/337821411_Predicting_Flight_Prices_in_India/links/5debfba992851c83646b669a/Predicting-Flight-Prices-in-India.pdf)
   - [Understanding Customer Perception While Booking Flight Tickets](http://www.solidstatetechnology.us/index.php/JSST/article/view/5721)
   - [Airline Fare Prediction Using Machine Learning Algorithms](https://ieeexplore.ieee.org/abstract/document/9716563)

## Business Understanding
Maskapai penerbangan x merupakan salah satu perusahaan maskapai yang menyediakan moda transportasi pesawat udara di Negara India. Untuk memaksimalkan keuntungan perusahaan, maka perusahaan harus mengetahui harga tiket ideal untuk diterapkan pada layanan jasa penerbangannya.

Harga tiket ideal adalah harga yang tidak terlalu mahal maupun tidak terlalu murah. Jika harga tiket terlalu mahal, maka akan membuat calon pengguna layanan berpikir beberapa kali untuk menggunakan layanan penerbangan di maskapai x. Namun jika terlalu murah, maka keuntungan perusahaan tidak akan maksimal, bahkan akan menderita kerugian, karena biaya operasional tidak sebanding dengan pendapatan. 

Oleh sebab itu, maka perlu dibuat aplikasi yang dapat memprediksi harga ideal tiket pesawat di India, untuk menjadi bahan pengambilan keputusan bagi pimpinan dalam menentukan harga layanan.

Dengan menerapkan harga tiket yang ideal, maka pengguna layanan akan merasa senang dan berpotensi untuk menambah pelanggan yang ingin menggunakan layanan penerbangan melalui maskapai penerbangan x.

### Problem Statements

Berdasarkan penjelasan yang telah disampaikan sebelumnya, maka problem statements (rumusan masalah), yaitu sebagai berikut:
- Apa faktor-faktor yang dapat mempengaruhi harga tiket pesawat?  
- Berapa harga ideal tiket pesawat untuk diterapkan di maskapai penerbangan x?

### Goals

Tujuan yang ingin dicapai dari pembuatan aplikasi prediksi harga tiket pesawat di India ini, yaitu sebagai berikut:
- Mengetahui faktor-faktor yang mempengaruhi harga tiket pesawat?
- Membuat aplikasi yang dapat memprediksi harga tiket pesawat secara akurat, sebagai bahan pengambilan keputusan dalam penerapan harga tiket ideal untuk diterapkan dimaskapai penerbangan x.

    ### Solution statements
    - Solusi yang dapat dilakukan untuk menangani permasalahan sebagaimana terdapat dalam problem statements, yaitu dengan membuat aplikasi prediksi harga tiket pesawat. Adapun aplikasi tersebut dibuat dengan menerapkan teknologi machine learning serta bahasa pemrograman python.
    - Algoritma machine learning yang akan digunakan, yaitu Random Forest dan Boosting Algorithm.
    - Untuk mengukur keakuratan/keidealan prediksi harga tiket pesawat yang dilakukan oleh aplikasi yang dibuat, maka metrik yang digunakan adalah Mean Squared Error (MSE). 

## Data Understanding
Data yang digunakan adalah dataset yang bersumber dari situs Kaggle yang berisi dataset terkait tiket pesawat di maskapai penerbangan India. Dataset sebagaimana dimaksud dapat didownload pada link berikut ini: [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction). Jumlah data yang terdapat didalam Flight Price Prediction sebanyak 300153 data.

### Variabel-variabel yang terdapat dalam dataset Flight Price Prediction:
- airline = Nama maskapai penerbangan.
- source_city = Kota awal pemberangkatan.
- departure_time = Waktu pemberangkatan.
- stops = jumlah transit selama perjalanan.
- arrival_time = Waktu tiba di kota tujuan.
- destination_city = Kota tujuan penerbangan.
- class = Kelas penerbangan.
- duration = Waktu yang dibutuhkan untuk tiba di kota tujuan.
- days_left = Jarak hari pemesanan tiket dengan hari penerbangan.
- price = harga tiket.

**Langkah-Langkah Dalam Data Understanding**:
- Untuk memahami dataset, langkah-langkah yang dilakukan, yaitu sebagai berikut:
    - Melakukan load dataset kedalam google colaboratory.
    - Melakukan Exploratory data analysis untuk memahami makna-makna variabel yang terdapat dalam dataset.
    - menggunakan teknik visualisasi data kategorik dan non kategorik dengan menggunakan library seaborn.
    - Memvisualisasikan data dengan menggunakan boxplot untuk mencari outlier.
    - Menggunakan IQR (Interquartile Range) untuk mengeliminasi outlier.
    - Melakukan univariative analysis untuk memahami sebaran data variabel.
    - Melakukan multivariative analysis untuk memahami korelasi variabel kategorikal dan numberikak terhadap variabel price.

## *Data Preparation*
Teknik *data preparation* yang dilakukan, yaitu sebagai berikut:
1. Mengubah dataset *flight price prediction* menjadi *dataframe* dengan menggunakan *pandas*.
2. Analisis awal untuk membuang variabel yang sangat tidak relevan untuk prediksi data.
3. Melakukan *exploratory data analysis* untuk memahami variabel-variabel yang terdapat dalam dataset.
4. Memvisualisasikan data dengan menggunakan *boxplot* untuk mencari *outlier*.
5. Menggunakan IQR *(Interquartile Range)* untuk mengeliminasi outlier.
6. Melakukan *univariative analysis* untuk memahami sebaran data variabel.
7. Melakukan *multivariative analysis* untuk memahami korelasi variabel kategorikal dan numberikal terhadap variabel *price*.  
8. Membuat *correlation matrix* untuk fitur numerik.
9. Mengeliminasi variabel numerik yang memiliki korelasi rendah terhadap variabel *price*.

**Hasil Exploratory Data Analysis**

![image](https://user-images.githubusercontent.com/122204998/215275455-f83febae-e954-485f-bd9b-afd26be84f18.png)

Gambar 1: visualisasi variabel duration dengan menggunakan boxplot

Berdasarkan hasil visualisasi, terlihat terdapat outliers dalam variabel duration. Alasannya, karena terdapat data yang terdapat melebihi Quartil 3.

![image](https://user-images.githubusercontent.com/122204998/215275641-df70eac8-b271-4935-b5ee-b4c28e2f21aa.png)

Gambar 2: visualisasi variabel days_left dengan menggunakan boxplot

Berdasarkan hasil visualisasi, tidak terdapat outliers dalam variabel days_left.

![image](https://user-images.githubusercontent.com/122204998/215275715-4455ce36-d97b-4919-b969-07ad60d44acc.png)

Gambar 3: univariate analysis pada variabel airline

Berdasarkan hasil visualisasi, maskapai penerbangan vistara memiliki jumlah data yang paling banyak dibandingkan dengan masakapai penerbangan lainnya, sedangkan maskapai penerbangan SpiceJet memiliki jumlah data yang paling sedikit dibandingkan dengan maskapai penerbangan lainnya.

![image](https://user-images.githubusercontent.com/122204998/215275873-b6947df9-135d-405c-b645-7733db19a332.png)

Gambar 4: univariate analysis pada variabel source_city

Berdasarkan hasil visualisasi, Delhi memiliki jumlah data yang paling banyak dibandingkan dengan kota keberangkatan lainnya, sedangkan Chennai memiliki jumlah data yang paling sedikit dibandingkan dengan kota keberangkatan lainnya.

![image](https://user-images.githubusercontent.com/122204998/215276030-75013578-016d-4d3b-94cf-da7fe78773d1.png)

Gambar 5: univariate analysis pada variabel departure_time

Berdasarkan hasil visualisasi, waktu keberangkatan pagi (morning) memiliki jumlah data yang paling banyak dibandingkan dengan kota keberangkatan lainnya, namun masih relatif seimbang, kecuali waktu keberangkatan tengah malam memiliki jumlah data yang sangat sedikit dibandingkan dengan waktu keberangkatan lainnya.

![image](https://user-images.githubusercontent.com/122204998/215276157-c79c42f4-555d-45c8-ab5c-b8197aaaf718.png)

Gambar 6: univariate analysis pada variabel destination_city

Berdasarkan hasil visualisasi, waktu keberangkatan pagi (morning) memiliki jumlah data yang paling banyak dibandingkan dengan kota keberangkatan lainnya, namun masih relatif seimbang, kecuali waktu keberangkatan tengah malam memiliki jumlah data yang sangat sedikit dibandingkan dengan waktu keberangkatan lainnya.

![image](https://user-images.githubusercontent.com/122204998/215276300-e37a7db7-8f6e-4e12-b292-c422cf019d08.png)

Gambar 7: univariate analysis pada variabel class

Berdasarkan hasil visualisasi, kelas ekonomi mendominasi jumlah data di dalam dataset, dibandingkan dengan kelas bisnis.

![image](https://user-images.githubusercontent.com/122204998/215276447-0869bef6-e328-41f0-9258-47fbe7260974.png)

Gambar 8: univariate analysis pada stops, duration, variabel days_left, price dengan menggunakan histogram

Berdasarkan hasil visualisasi, maka dapat dibuat kesimpulan sebagai berikut:
- kebanyakan pesawat melakukan transit sebanyak satu kali.
- semakin cepat waktu penerbangan, maka harga tiket pesawat semakin mahal.
- variabel days_left tidak begitu berpengaruh pada harga tiket pesawat.

![image](https://user-images.githubusercontent.com/122204998/215325509-8f151876-8200-4db2-b8fc-77759423de7d.png)

Gambar 9: multivariate analysis antara variabel airline dengan price

Berdasarkan hasil visualisasi, variabel airline memiliki korelasi yang kuat terhadap price.

![image](https://user-images.githubusercontent.com/122204998/215325764-55e8913b-14f3-49b2-8b07-6646cc08a379.png)

Gambar 10: multivariate analysis antara variabel source_city dengan price

Berdasarkan hasil visualisasi, variabel source city kurang memiliki korelasi terhadap price.

![image](https://user-images.githubusercontent.com/122204998/215325834-fae3b485-55bf-4e11-b61a-60a3cdd993b0.png)

Gambar 11: multivariate analysis antara variabel departure_time dengan price

Berdasarkan hasil visualisasi, variabel departure_time cukup memiliki korelasi terhadap price.

![image](https://user-images.githubusercontent.com/122204998/215325927-c2e1f93b-f6de-4eb4-adf4-ddf382d952cb.png)

Gambar 12: multivariate analysis antara variabel arrival_time dengan price

Berdasarkan hasil visualisasi, variabel arrival_time cukup memiliki korelasi terhadap price.

![image](https://user-images.githubusercontent.com/122204998/215326002-60fecf19-efc5-4213-813b-ab022cff7bef.png)

Gambar 13: multivariate analysis antara variabel destination_city dengan price

Berdasarkan hasil visualisasi, variabel destination_city kurang memiliki korelasi terhadap price.

![image](https://user-images.githubusercontent.com/122204998/215326095-7fde0672-65cd-4773-b75a-870225f74455.png)

Berdasarkan hasil visualiasi, variabel class memiliki korelasi yang kuat terhadap price.

![multivariate numerical variable](https://user-images.githubusercontent.com/122204998/215327836-cbe9e62d-b87d-499c-a7df-052ae29507f1.gif)

Gambar 13: multivariate analysis variabel stops, duration, days_left terhadap price

Berdasarkan hasil visualiasi, dapat dibuat kesimpulan bahwa variabel stops, duration dan days_left kurang memiliki korelasi dengan price.

![correlation matrix](https://user-images.githubusercontent.com/122204998/215327123-a9d84239-f9c7-467a-b853-7eceb68719fe.jpg)

Gambar 14: Correlation Matrix

Bila kita lihat, bahwa korelasi variabel stops terhadap price = 0.12, korelasi variabel duration terhadap price = 0.22, sedangkan korelasi variabel days_left terhadap price sebesar 0.09.

**Proses *Data Preparation***: 
- Proses data preparation dilakukan melalui langkah-langkah, yaitu sebagai berikut: Melakukan *load* data pada *google colaboratory*, kemudian melakukan analisis awal terkait variabel yang sangat tidak relevan untuk diproses lebih lanjut. Selanjutnya, memahami makna-makna variabel dengan menerapkan *Exploratory Data Analysis*, kemudian melakukan visualisasi data untuk mencari outlier dengan menggunakan *boxplot* dari *library seaborn*. Selanjutnya, menerapkan metode IQR untuk mengeliminasi outlier, kemudian menggunakan *univariate analysis* serta *multivariative analysis*. Selanjutnya membuat *correlation matrix*, kemudian membuang variabel numberik yang memiliki korelasi rendah terhadap variabel *price.*
- Data preparation diperlukan agar data yang akan diproses oleh algoritma *machine learning* bebas dari *outlier* dan variabel-variabel yang digunakan untuk algoritma adalah variabel yang memiliki korelasi tinggi terhadap penentuan prediksi harga tiket pesawat.
- Pembuatan aplikasi ini menggunakan IQR *(Interquartile Range)* untuk mengeliminasi *outlier* yang terdapat dalam dataset *flight price prediction*.

## Modeling
- Model *machine learning* yang digunakan adalah *random forest* dan *boosting algorithm*.
- *Random forest* adalah model *machine learning* yang menerapkan gabungan model *machine learning decision tree*.
- Untuk menggunakan model *Random Forest* menggunakan *function RandomForestRegressor* yang merupakan bagian dari *library sklearn.ensemble*.
- Parameter yang digunakan dalam model *Random Forest*, yaitu sebagai berikut:
    - *n\_estimators* = jumlah pohon keputusan *(decision tree)* yang akan dibuat pada model *Random Forest* yang digunakan. Pada model ini n_estimators yang di buat, yaitu 100.
    - *max_depth* = maksimal kedalaman dari decision tree yang akan dibuat. Pada model Random Forest ini kedalaman yang dibuat sampai 64 level.
    - *max_features* = feature maksimal yang digunakan ketika melakukan split. Parameter yang di input adalah sqrt.
    - *random_state* = mengatur status random dari model *Random Forest*. Pada model ini *random state* yang digunakan adalah 1.
    - *n\_jobs* = mengatur sistem paralel dan penggunaan prosesor dalam proses decision tree. Pada model *Random Forest* ini, paramater di isi -1 agar seluruh prosesor digunakan.
    - *verbose* = mengatur tampilan pada saat proses *training*. Nilai yang di input adalah 2, sehingga hasil dari *training* ditampilkan setiap langkah.
    - *warm_start* = mengatur apakah *weight* hasil pelatihan sebelumnya akan digunakan lagi pada *training* baru atau tidak. Nilai yang di input adalah *True*.
    - *RF.fit(X_train, y_train)* = menentukan data yang akan digunakan pada proses *training* model *Random Forest*.
    - *models.loc* = mengakses kolom dan baris dari *dataframe* yang digunakan untuk proses *training* model *Random Forest*
    - *mean_absolute_error* = metrik yang digunakan untuk mengukur akurasi model yang telah dilatih.
 - Demi mendapatkan hasil yang terbaik, selain menggunakan random forest, digunakan algoritma lain, yaitu ***boosting algorithm*** sebagai algoritma pembanding untuk mengukur manakah algoritma yang lebih baik diantara keduanya dalam menghasilkan prediksi harga tiket pesawat.
 - *Boosting algorithm* adalah salah satu algoritma *ensemble learning* yang proses laltihannya dilakukan secara sekuensial serta iteratif.
 -  Parameter yang digunakan dalam *boosting algorithm*, yaitu sebagai berikut:
	 - *AdaBoostRegressor* = function yang digunakan untuk melakukan proses training model dengan menggunakan *Boosting Algorithm*. Function ini berada pada library / modul *sklearn.ensemble*.
	 - *learning_rate*= parameter yang digunakan untuk mengatur proses training dari algoritma ini. Pada model ini, paramater di isi 1.0.
	 - *random_state* = mengatur status random dari model *Boosting Algorithm*. Pada model ini *random state* yang digunakan adalah 1.
	 - *boosting.fit(X_train, y_train)* = load data yang akan digunakan dalam *training* model *Boosting Algorithm*.
	 - *models.loc['train_mse','Boosting']* = mengakses kolom dan baris dari *dataframe* yang digunakan untuk proses *training* model *Boosting Algorithm*.
	 - *mean_absolute_error* = metrik yang digunakan untuk mengukur akurasi model yang telah dilatih.
	 
**Kelebihan dan Kekurangan Random Forest dan Boosting Algorithm** 
- Setelah melakukan training menggunakan *Random Forest* dan *Boosting Algorithm*, maka dapat disimpulkan kelebihan dan kekurangan masing-masing, yaitu sebagai berikut:
- Kelebihan algoritma *Random Forest*, yaitu dapat hasil prediksi masih andal meskipun ada *noise* maupun *missing value* pada dataset yang digunakan.
- Kekurangan *Random Forest* , yaitu untuk mendapatkan prediksi yang akurat, *tuning* parameter harus dilakukan secara tepat.
- Kelebihan *Boosting Algorithm*, yaitu memori untuk proses latihan model relatif lebih kecil dibandingkan dengan Random Forest. 
- Kekurangan *Boosting Algorithm*, yaitu sensitif pada *noise* dan *missing value*, apabila dibandingkan dengan *Random Forest*.
- Berdasarkan hasil training model, maka ditetapkan bahwa algoritma yang terbaik diantara *Random Forest* dan *Boosting Algorithm* dalam memprediksi harga tiket, yaitu algoritma *Random Forest*.
- Alasannya, karena nilai *Mean Absolute Error (MAE)* yang dihasilkan *Random Forest* lebih baik dari *Boosting Algorithm*.

## *Evaluation*
Tabel 1: Hasil Evaluasi Model dengan Menggunakan Mean Absolute Error
Model                        | train       | test	  |
---------------------------- | ----------- | ------------ |
Random Forest (RF)           | 2009.858278 | 2218.153604  |
Boosting Algorithm (Boosting)| 4818.380262 | 4790.128881  |
- Metrik yang digunakan untuk mengukur hasil *training* adalah *mean absolute error (MAE)*. 
- Berdasarkan hasil training, bahwa model *Random Forest* menghasilan nilai MAE pada saat *training* = 2009.8582782434323 dan pada saat tes = 2218.153604433864. 
- Ketika dianalisis, nilai MAE tersebut kurang dari 10%, sehingga model sudah dapat dikatakan menghasilkan nilai yang baik *(good fit)*.
- Berdasarkan hasil training model, maka ditetapkan bahwa algoritma yang terbaik diantara *Random Forest* dan *Boosting Algorithm* dalam memprediksi harga tiket, yaitu algoritma *Random Forest*.\
- Alasannya, karena nilai *Mean Absolute Error (MAE)* yang dihasilkan *Random Forest* lebih baik dari *Boosting Algorithm*.

**Cara Kerja Metrik Mean Absolute Error**: 
- *Mean Absolute Error* adalah metrik statistik yang digunakan untuk mengukur keakuratan dari prediksi nilai yang bersifat kontinyu.
- Semakin kecil nilai MAE, maka akan semakin baik pula model tersebut dalam melakukan prediksi nilai.

**Kesimpulan**
- Berdasarkan hasil training dan test, maka algoritma yang terbaik adalah *Random Forest*, alasannya karena nilai *Mean Absolute Error (MAE)* yang dihasilkan *Random Forest* lebih baik dari *Boosting Algorithm*.
- Model telah good fit dalam melakukan akurasi, alasannya karena nilai MAE yang dihasilkan kurang dari 10% 2009.858278 sedangkan 10% nilai MAE, yaitu 9786.7
- 10 % nilai MAE dihitung dengan rumus: mae_target = (flight['price'].max() - flight['price'].min()) * 10/100 
