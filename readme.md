# Laporan Proyek Machine Learning - Abdul Aziz Munawar

## Project Overview
Film merupakan sarana hiburan yang digemari banyak orang dalam mengisi waktu luang. Untuk memberikan pengalaman terbaik bagi *User* situs *Web Streaming Video Online*, maka diperlukan sistem yang mampu memberikan rekomendasi-rekomendasi film menarik sesuai minat user. Oleh sebab itu, dalam proyek ini saya mengembangkan suatu sistem yang mampu memberikan rekomendasi film menarik sesuai minat *User* dengan metode *Content Based Filtering* dan *Collaborative Filtering*.

Dengan metode *Content Based Filtering*, User akan diberikan rekomendasi film berdasarkan kesamaan genre dari film yang pernah ditontonnya. Jika User *Web Streaming Video Online* pernah menonton film dengan genre *action* (laga), maka sistem akan memberikan rekomendasi film bergenre *action* juga. Selanjutnya, jika user pernah menonton film dengan genre *horror*, maka sistem akan memberikan rekomendasi film bergenre horror juga.

Dengan metode Collaborative Filtering, sistem akan memberikan rekomendasi kepada User berdasarkan kesamaan perilaku seorang user dengan user lainnya. Jika perilaku User A mirip dengan perilaku User B, maka User A akan diberikan rekomendasi film yang mirip dengan User B. Untuk mencari kesamaan perilaku tersebut, pada proyek ini kita akan menggunakan rating yang diberikan oleh User pada film yang pernah ditontonnya, kemudian mencari kesamaan perilaku tersebut dengan user lainnya. 

Dengan adanya sistem rekomendasi ini, diharapkan dapat memberikan pengalaman terbaik bagi User dalam menonton video di Situs *Web Streaming Video Online*. User yang merasa puas terhadap fitur yang tersedia dalam website, akan terus berlangganan terhadap website tersebut.

**Alasan Pembuatan Proyek dan Referensi Terkait Proyek**:
- Proyek Sistem rekomendasi ini sangat penting untuk dikembangkan. Alasannya, Untuk memberikan pengalaman terbaik bagi User situs Web Streaming Video Online, maka diperlukan sistem yang mampu memberikan rekomendasi-rekomendasi film menarik sesuai minat user. User yang merasa puas terhadap fitur yang tersedia dalam website, akan terus berlangganan terhadap website yang dimiliki.
- Referensi yang relevan terkait proyek sistem rekomendasi ini, yaitu sebagai berikut:
  - [Sistem Rekomendasi Film Menggunakan Metode User Based Collaborative Filtering](https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/16513) 
  - [Movie Recommender Systems: Concepts, Methods, Challenges, and Future Directions](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9269752/)
  - [Performance Evaluation of Recommender Systems](http://www.ijpe-online.com/EN/abstract/abstract3798.shtml)
  - [Can Automated Group Recommender Systems Help Consumers Make Better Choices?](https://journals.sagepub.com/doi/abs/10.1509/jm.10.0537?journalCode=jmxa)

## Business Understanding

*Web Streaming Video Online* adalah website yang menyediakan layanan menonton video streaming secara online. Terdapat banyak film yang tersedia pada website tersebut untuk ditonton oleh para User yang berlangganan (subscriber). Untuk memberikan pengalaman terbaik bagi User dalam menonton video di situs tersebut, maka diperlukan sistem yang mampu merekomendasikan film-film yang menarik sesuai minat User tersebut.

Memberikan pengalaman terbaik bagi User merupakan hal yang sangat penting. Alasannya, karena Web Streaming Video Online tidak hanya satu. Terdapat Web Streaming Video Online kompetitor yang berusaha untuk meningkatkan pelayanan, agar websitenya banyak memiliki User yang berlangganan. Jika Web Streaming Video Online kalah oleh kompetitor, maka User yang semula berlangganan, kemungkinan besar akan berpindah ke website kompetitor. Hal tersebut merupakan suatu permasalahan yang harus diselesaikan.

Oleh sebab itu, pada proyek ini akan dibuat sistem yang mampu memberikan rekomendasi film-film menarik sesuai minat User, baik berdasarkan metode Content Based Filtering dan Collaborative Filtering.

Content Based Filtering adalah sistem yang memberikan rekomendasi berdasarkan kategori yang ditentukan. Pada proyek ini, kategori yang akan dipakai sebagai acuan Content Based Filtering dalah genres.

Collaborative Filtering adalah sistem yang memberikan rekomendasi film berdasarkan kesamaan perilaku seorang user dengan user lainnya.

### Problem Statements

Berdasarkan uraian tersebut, maka rumusan masalah yaitu sebagai berikut:
- Bagaimana cara membuat sistem rekomendasi film dengan metode Content Based Filtering?
- Bagaimana cara membuat sistem rekomendasi film dengan metode Collaborative Filtering?

### Goals

Berdasarkan rumusan masalah tersebut, maka tujuan yang ingin dicapai yaitu sebagai berikut:
- Membuat sistem rekomendasi film dengan metode Content Based Filtering.
- Membuat sistem rekomendasi film dengan metode Collaborative Filtering.

**Solution Approach:**
Untuk membuat sistem rekomendasi film dengan metode Content Based Filtering dan Collaborative Filtering, kita akan menggunakan bahasa pemrograman Python, Google Colaboratory serta Module-Module Python yang relevan dengan proyek yang sedang dibuat, diantaranya yaitu pandas (untuk membuat, mengolah atau menghapus dataframe), numpy (untuk melakukan perhitungan matematis), tensorflow (untuk mengembangkan sistem berbasis machine learning) Matplotlib dan seaborn (untuk visualisasi data).

    ### Solution statements
    Sistem rekomendasi film ini menggunakan dua pendekatan/metode, yaitu metode Content Based Filtering dan metode Collaborative Filtering.
    Metode Content Based Filtering adalah metode yang memberikan rekomendasi berdasarkan kategori yang ditentukan. Pada proyek ini, kategori yang akan
    dipakai sebagai acuan Content Based Filtering dalah genres.
    
    Metode Collaborative Filtering adalah metode yang memberikan rekomendasi berdasarkan kesamaan perilaku seorang user dengan user lainnya.
    
## Data Understanding
Dataset yang digunakan dalam pembuatan sistem rekomendasi film dengan metode Content Based Filtering dan Collaborative Filtering, yaitu IMDb Movie Reviews Dataset. Dataset ini memiliki tiga file csv (comma separated value), yaitu imdb_1000.csv, movies.csv dan ratings.csv. Dari tiga file csv tersebut, terdapat dua file csv yang digunakan, yaitu movies.csv dan ratings.csv.

Dataset movies.csv memiliki 9.125 data judul film. Dataset ratings.csv memiliki 100.836 data rating yang berasal dari 610 user. Data tersebut diunduh dari website Kaggle melalui link berikut ini: [IMDb Movie Reviews Dataset](https://www.kaggle.com/datasets/hariprabu/imdb-1000?select=imdb_1000.csv).

Variabel-variabel yang terdapat dalam dataset movies.csv, yaitu sebagai berikut:
- movieId : berisi data mengenai nomor ID unik dari suatu judul film.
- title : berisi data mengenai judul film.
- genres : berisi data mengenai genre dari suatu judul film.

Variabel-variabel yang terdapat dalam dataset ratings.csv, yaitu sebagai berikut:
- userId : berisi data mengenai nomor ID unik dari user pengguna website IMDb.
- movieId : berisi data mengenai nomor ID unik dari suatu judul film yang dapat dihubungkan ke movieId pada movies.csv.
- rating : berisi penilaian user terhadap kualitas suatu judul film.
- timestamp : berisi data waktu user tersebut memberikan rating pada suatu judul film.

**Exploratory Data Analysis**:
- Untuk memastikan data pada daset tidak terdapat missing value, maka dilakukan pemeriksaan missing value. Berdasarkan hasil pemeriksaan, tidak terdapat missing value pada dataset movies.csv dan ratings.csv

![EDA movies.csv](https://user-images.githubusercontent.com/122204998/235833486-406885be-0187-439c-a662-47f3af3c798b.png)

- Berdasarkan gambar tersebut, dapat kita lihat bahwa tidak terdapat missing value pada variabel movieId, title maupun genres.

![EDA ratings.csv](https://user-images.githubusercontent.com/122204998/235833657-64289710-ec7b-4367-8fea-f0d22d77197e.png)

- Berdasarkan gambar tersebut, dapat kita lihat bahwa tidak terdapat missiong value pada variabel userId, movieId, rating dan timestamp.
- Dua file csv ini digabungkan menjadi satu dataframe, agar dapat dilakukan pemrosesan data sebagai input data untuk sistem rekomendasi Content Based Filtering dan Collaborative Filtering.
- Setelah dilakukan penggabungan file movies.csv dan ratings.csv, muncul missing value. Hal ini, terjadi karena perbedaan jumlah data pada tiap file csv.
- Untuk mengatasi missing value, maka data-data yang memiliki missing value akan dibersihkan dengan menggunakan perintah dropna.
- Untuk membersihkan data duplikat hasil dari penggabungan dua file csv, maka data duplikat tersebut dibersihkan dengan perintah drop_duplicates.
- Dataset bersih hasil penggabungan menjadi 7.072 baris dan terdiri dari 6 variabel (kolom), yaitu movieId, title, genres, userId, rating dan timestamp.
- Berdasarkan hasil analisis, dapat kita lihat bahwa variabel rating memiliki nilai minimal sebesar 0.5 dan nilai maksimal 5.0.
![Analisis Dataset](https://user-images.githubusercontent.com/122204998/235835189-758f157b-8fac-4305-a706-2ef1fd5a202f.png)
- Berdasarkan hal tersebut, maka rentang rating yang diberikan oleh user berada pada range 0.5 - 5.0 (digunakan pada collaborative filtering).
- Mencari outliers pada dataset. Berdasarkan hasil visualisasi data, tidak terdapat data outliers pada dataset. Hal ini bisa terlihat pada box plot berikut ini:

  ![Boxplot](https://user-images.githubusercontent.com/122204998/235888741-1d3a67ef-1337-49cc-bb21-7fa8340744f0.png)

- Variabel yang dianalisis dengan menggunakan Boxplot hanya variabel rating saja. Alasannya, karena variable rating memungkinkan adanya penilaian yang bias terhadap kualitas/rating suatu film yang diberikan oleh user.
- Untuk melihat komposisi data pada variabel rating, maka dilakukan visualisasi data dengan menggunakan diagram batang, yaitu sebagai berikut:

![Histogram](https://user-images.githubusercontent.com/122204998/235890901-fcbe176d-372e-4321-8dec-c4e0a5795de3.png)

- Berdasarkan hasil analisis, bahwa rating 4.0 adalah rating yang paling banyak diberikan oleh user disusul rating 3.0 diperingkat kedua. Selanjutnya rating 0.5 adalah rating yang paling sedikit diberikan oleh user disusul oleh rating 1.5.  

## Data Preparation
- Untuk menjamin dataset yang berkualitas, maka dataset telah dilakukan pemeriksaan dari kemungkinan adanya data missing value.
- Data missing value yang terdapat dalam dataset telah dibersihkan dengan perintah dropna
- Untuk membersihkan data duplikat hasil dari penggabungan dua file csv, maka data duplikat tersebut dibersihkan dengan perintah drop_duplicates.
- Dataset bersih hasil penggabungan menjadi 7.072 baris dan terdiri dari 6 variabel (kolom), yaitu movieId, title, genres, userId, rating dan timestamp.
- Dataset telah dianalisis dengan menggunakan metode Exploratory Data Analysis (EDA).
- Dataset untuk input sistem rekomendasi Content Based Filtering telah diseleksi dengan hanya menggunakan variabel yang relevan, yaitu movieId, title dan genres.
- Tiga variabel tersebut dibuat dataframe baru yang berperan sebagai data input pada sistem rekomendasi Content Based Filtering yang kita buat.
- Untuk proses training, data divektorisasi dengan menggunakan TF-IDF (Term Frequency Inverse Document Frequency). TF-IDF adalah suatu algoritma untuk mencari kata-kata yang dianggap penting dalam suatu dokumen.
- Selanjutnya, data diubah menjadi matriks agar dapat dihitung derajat kesamaannya (cosine similarity).
- Hasil dari cosine similarity digunakan sebagai data acuan sistem rekomendasi Content Based Filtering.
- Untuk sistem rekomendasi Collaborative Filtering, data diolah dengan cara di encoding terlebih dahulu.
- Dataset dibagi menjadi 80% untuk data training serta 20% untuk data validasi dalam proses pelatihan Collaborative Filtering.

**Penjelasan Proses Data Preparation dan Alasan Diperlukannya Data Preparation**: 
- Data preparation yang dilakukan meliputi: pemeriksaan missing value, menangani missing value dengan menggunakan perintah dropna. Selanjutnya, dilakukan pula pembersihan data duplikat dengan menggunakan perintah drop_duplicates pada dataset yang digunakan. Kemudian, untuk menghindari bias, maka dataset dipe Yulriksa dengan menggunakan visualisasi box plot pada variabel rating. Agar data yang diinput relevan dengan kebutuhan sistem Rekomendasi Content Based Filtering, maka variabel yang dijadikan input diseleksi, sehingga hanya menggunakan variabel movieId, title dan genres.Sebelum dilakukan proses training data divektorisasi dengan TF-IDF. Setelah divektorisasi, data diubah kedalam bentuk matriks agar dapat dilakukan perhitungan derajat kesamaan suatu film dengan film lainnya dilihat dari segi genre.
- Selanjutnya, khusus untuk data preparation sistem rekomendasi Collaborative Filtering, data di encoding terlebih dahulu serta di split 80% untuk data training dan 20% untuk data validasi.

- Dapat kita lihat, bahwa kita telah melakukan serangkaian proses dalam data preparation. Namun, mengapa hal tersebut diperlukan? Data preparation penting dilakukan karena dataset awal yang terdapat dalam file csv masih bersifat mentah, sehingga perlu diolah agar yang dijadikan input pada sistem rekomendasi relevan dan memiliki kualitas data yang baik.

## Modeling
- Sistem rekomendasi film yang dibuat menggunakan Metode *Content Based Filtering* dan *Collaborative Filtering*. Dengan metode *Content Based Filtering*, User akan diberikan rekomendasi film berdasarkan kesamaan genre dari film yang pernah ditontonnya. Jika User Web Streaming Video Online pernah menonton film dengan genre *action* (laga), maka sistem akan memberikan rekomendasi film bergenre *action*. Selanjutnya, jika User pernah menonton film dengan genre *horror*, maka sistem akan memberikan rekomendasi film bergenre *horror*.
- Pemodelan Sistem rekomendasi film *Content Based Filtering* menggunakan *function* yang ditulis dalam bahasa *Python*.
- Penjelasan mengenai function tersebut, yaitu sebagai berikut:
  - *def movie_recommendation_CBF* = Membuat function dengan nama movie_recommendation_CBF.
  - *(title, similarity_data=cosine_sim_df, items=movie_content[['title', 'genres']], k=5)* = parameter yang digunakan pada function tersebut, terdapat empat           parameter, yaitu title (judul film yang ingin dicari rekomendasinya), similarity_data (derajat kesamaan dari film yang ingin dicari), items (menampilkan data judul dan genre dari film), k (jumlah rekomendasi film yang ingin ditampilkan).
  -  index = variabel untuk menampung data hasil sorting dan partisi data film.
  -  closest = variabel untuk menampung data film yang memiliki nilai derajat kesamaan paling besar dari film yang ingin dicari rekomendasinya.
  -  return pd.DataFrame(closest).merge(items).head(k) = membuat dataframe dari variabel closest sebanyak 5 (nilai dari k). 

- Untuk melihat hasil yang diberikan Sistem Rekomendasi Film Content Based Filtering dalam proyek ini, dapat menggunakan perintah sebagai berikut: 
  movie_recommendation_CBF('Judul Film')

- Contoh:
  movie_recommendation_CBF('Ace Ventura: When Nature Calls (1995)')

Hasil sistem rekomendasi, yaitu sebagai berikut:

![Content Based Filtering](https://user-images.githubusercontent.com/122204998/236146412-06b6fac1-9d07-46bc-bf4c-4f05ff435150.png)

- Film Ace Ventura: When Nature Calls (1995) adalah film yang memiliki genre Comedy. Bila kita lihat, bahwa rekomendasi 5 film yang diberikan juga merupakan film yang bergenre sama, yaitu Comedy.

- Kemudian, sistem rekomendasi dengan metode Collaborative Filtering akan memberikan rekomendasi kepada User berdasarkan kesamaan perilaku seorang user dengan user lainnya. Jika perilaku User A mirip dengan perilaku User B, maka User A akan diberikan rekomendasi film yang mirip dengan User B.
- Pemodelan sistem rekomendasi Collaborative Filtering ini menggunakan bahasa pemrograman Python.
- Model dicompile dengan Binary Crossentropy untuk melakukan perhitungan loss function
- Optimizer yang digunakan dalam model ini adalam Adam.
  Metriks yang digunakan untuk menghitung akurasi sistem rekomendasi adalah Root Mean Squared Error (RMSE). 
- Dataset dibagi menjadi dua, yaitu 80% untuk data training serta 20% untuk data validasi.
- Training dilakukan sebanyak 100 epoch dengan batch size per epoch adalah 64.
- Film yang akan direkomendasikan merupakan film baru yang sebelumnya belum pernah ditonton oleh user tersebut.
- Film yang akan direkomendasikan merupakan top 10 film yang mungkin akan membuat tertarik user tersebut. Rekomendasi ini berasal dari rating yang diberikan oleh user tersebut di Web Streaming Video Online. Kemudian sistem akan mencari kesamaan perilaku seorang user dengan user lainnya. Jika perilaku User A mirip dengan perilaku User B, maka User A akan diberikan rekomendasi film yang mirip dengan User B.
- Berikut ini 10 film yang direkomendasikan dengan menerapkan *Collaborative Filtering*, yaitu sebagai berikut:

  ![Collaborative Filtering](https://user-images.githubusercontent.com/122204998/236192234-5b366214-cbe5-4263-97cb-4612e4d8d13a.png)
  
**Kelebihan dan Kekurangan Content Based Filtering dan Collaborative Filtering**: 
- Untuk membuat sistem rekomendasi yang mampu memberikan rekomendasi yang tepat pada User, maka dalam proyek ini dibuat sistem rekomendasi yang menerapkan dua metode/algoritma, yaitu metode Content Based Filtering dan Collaborative Filtering.
- Dengan metode *Content Based Filtering*, User akan diberikan rekomendasi film berdasarkan kesamaan genre dari film yang pernah ditontonnya.
- Kelebihan dari metode *Content Based Filtering* adalah metode ini dapat memberikan rekomendasi film-film yang baru dirilis kepada User lebih cepat, tanpa terlebih dahulu menunggu penilaian/rating User lain. Alasannya, karena sistem rekomendasi ini memberikan rekomendasi sesuai dengan kesamaan genre antar suatu film, tanpa terpengaruh penilaian/rating user lain.
- Kelemahaan dari metode *Content Based Filtering* adalah setiap judul harus dilabel genre secara tepat. Jika ada judul yang sebenarnya memiliki genre yang sama, tapi karena kesalahan input label, sehingga diberi label genre yang berbeda, maka sistem rekomendasi ini kemungkinan besar tidak akan merekomendasikan film yang salah label tersebut. Selain dari itu, metode *Content Based Filtering* cenderung memberikan sistem rekomendasi yang monoton (tidak ada efek surprise), karena kemungkinan besar hanya akan merekomendasikan film yang bergenre itu-itu saja.
- Kelebihan dari metode *Collaborative Filtering* adalah metode ini dapat memberikan rekomendasi film-film tanpa melabeli film terlebih dahulu, karena sistem rekomendasi akan mencari kesamaan perilaku antara suatu user dengan user lainnya. Selain dari itu, sistem rekomendasi ini dapat memberikan efek surprise, karena memberikan rekomendasi film dengan genre berbeda, namun tetap memperhatikan minat User.
- Kelemahan dari metode *Collaborative Filtering*, yaitu sistem membutuhkan banyak input/feedback dari User agar sistem dapat berfungsi secara baik.
- Berdasarkan uraian tersebut, dengan menggabungkan dua metode sistem rekomendasi (Content Based Filtering dan Collaborative Filtering), maka metode tersebut dapat melengkapi serta menambah keberagaman dalam pemberian rekomendasi film bagi User.  

## Evaluation
- Untuk mengukur ketepatan sistem rekomendasi metode Content Based Learning dalam memberikan rekomendasi film, kita dapat menggunakan precision. Untuk mengetahui precision dari sistem rekomendasi, kita dapat menghitung dengan rumus: (jumlah rekomendasi yang sesuai genre/jumlah total rekomendasi film) * 100%.
- Berdasarkan hasil pemeriksaan, bahwa ketika sistem diperintahkan untuk mencari rekomendasi film **Ace Ventura: When Nature Calls (1995)** yang bergenre **comedy**, sistem mampu memberikan 5 rekomendasi film yang bergenre comedy juga.
- Dengan memasukan rumus: jumlah rekomendasi yang sesuai genre/jumlah total rekomendasi film, maka: (5/5) * 100% = 100%, sehingga dapat dibuat kesimpulan bahwa sistem rekomendasi sudah baik (Good Fit).
- Untuk mengukur ketepatan sistem rekomendasi metode Collaborative Filtering, metrik evaluasi yang digunakan untuk mengukur ketepatan sistem rekomendasi dalam memberikan rekomendasi film, yaitu menggunakan Root Mean Squared Error (RMSE).
- RMSE adalah metrik statistik yang digunakan untuk mengukur keakuratan dari prediksi suatu nilai (dalam hal ini nilai kesamaan perilaku user).
- Semakin kecil nilai RMSE, maka akan semakin baik pula model tersebut dalam memberikan rekomendasi.
- Nilai RMSE yang didapatkan pada hasil pelatihan model setelah 100 epoch, yaitu:

Metrik                        | Nilai   |
----------------------------- | ------- |
root_mean_squared_error       | 0.1948  |
val_root_mean_squared_error   | 0.2058  |

- Secara visualisasi dapat digambarkan sebagai berikut:

![visualisasi hasil pelatihan](https://user-images.githubusercontent.com/122204998/236227317-c04a07fa-716b-4682-b3fe-1f84d7cceeaf.png)


-  Berdasarkan nilai RMSE yang didapatkan, maka dapat dibuat kesimpulan bahwa sistem rekomendasi telah Good Fit (baik). Alasannya, karena nilai RMSE nya memiliki nilai error dibawah 10%. Nilai RMSE error 10% adalah 0.45, sedangkan nilai RMSE yang didapatkan dibawah 0.45. 

**Cara Kerja Metrik Root Mean Squared Error (RMSE)**: 
- RMSE adalah metrik statistik yang digunakan untuk mengukur keakuratan dari prediksi suatu nilai (dalam hal ini nilai kesamaan perilaku user).
- Semakin kecil nilai RMSE, maka akan semakin baik pula model tersebut dalam memberikan rekomendasi.
- Nilai RMSE error 10% adalah 0.3029659363385887, sedangkan nilai RMSE error yang didapat pada hasil pelatihan model, yaitu 0.1948 (pada saat training) dan 0.2058 (pada saat validasi).

**Kesimpulan**
- Proyek ini telah berhasil membuat sistem rekomendasi dengan metode Content Based Filtering dan Collaborative Filtering.
- Dengan menggabungkan dua metode sistem rekomendasi (Content Based Filtering dan Collaborative Filtering), maka metode tersebut dapat melengkapi serta menambah keberagaman dalam pemberian rekomendasi film bagi User. 
- Sistem rekomendasi yang dibuat pada proyek ini dapat dikategorikan Good Fit, hal ini dapat dilihat rekomendasi yang diberikan sistem sudah memenuhi kriteria sistem rekomendasi yang baik, yaitu pada Content Based Filtering (sistem telah mampu memberikan rekomendasi film dengan genre yang sama) dan pada Collaborative Filtering (sistem telah mampu menghasilkan nilai RMSE yang rendah dibawah 10%).
- Rumus untuk menghitung 10% RMSE:
  - rmse_target = sqrt(mean_squared_error(y, ratings['rating']))
  - rmse_target = 0.1 * rmse_target
  - rmse_target
