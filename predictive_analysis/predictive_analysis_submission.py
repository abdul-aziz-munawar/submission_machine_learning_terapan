# -*- coding: utf-8 -*-
"""Predictive_Analysis_Submission.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MeiGI-UCPD-p_EUY24iHHHc-bN2Orz74

**Import library dan module yang dibutuhkan**

---
Mengimpor library dan module yang dibutuhkan untuk membuat project ini
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import seaborn as sns

from google.colab import drive
from google.colab import data_table
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

"""**Install library Kaggle**

---

Menginstal library tambahan yang dibutuhkan. Dalam hal ini, library Kaggle yang nantinya akan digunakann untuk load dataset dari situs Kaggle
"""

#install library dari luar
!pip install kaggle

"""**Mount Google Drive Ke Dalam Google Colab**

---
Memberikan izin akses Google Colab untuk menggunakan Google Drive pribadi. Agar dataset yang telah di download dari situs Kaggle tidak hilang saat session timeout, maka dataset tersebut akan disimpan pada Google Drive pribadi.

"""

drive.mount('/content/gdrive')

"""**Pengatur Path Penyimpanan Dataset**

---
Mengatur path (direktori/folder) yang akan digunakan dalam penyimpanan dataset.
"""

# Commented out IPython magic to ensure Python compatibility.
os.environ['KAGGLE_CONFIG_DIR'] = '/content/gdrive/MyDrive/Dataset'
# %cd /content/gdrive/MyDrive/Dataset

"""**Download Dataset**

---
Download dataset dari situs Kaggle untuk digunakan dalam project ini

"""

!kaggle datasets download -d shubhambathwal/flight-price-prediction

"""**Ekstrak Dataset**

---
Ekstrak dataset yang masih dalam file .zip

"""

!unzip \*.zip && rm *.zip

"""**Load Dataset dan Proses Cleaning Awal**

---
Dataset yang telah tersedia di Google Drive kemudian di load. Setelah mengamati beberapa kolom yang terdapat dalam dataset. Ada dua kolom yang dapat langsung dieliminasi, karena tidak relevan dengan proses analisis data.

"""

flight = pd.read_csv('/content/gdrive/MyDrive/Dataset/Clean_Dataset.csv')
flight.drop(['Unnamed: 0', 'flight'], axis = 1, inplace = True)
flight

"""**Deskripsi kolom yang terdapat dalam dataset**

---
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

**Memeriksa Missing Value**

---
Memeriksa dataset untuk mengetahui apakah dataset memiliki missing value. Berdasarkan hasil analisis, tidak terdapat missing value.
"""

flight.info()

"""**Mengubah satuan nilai dalam kolom duration**

---
Mengubah satuan nilai dalam kolom duration dari jam ke menit
"""

flight['duration'] = flight['duration'].apply(lambda x: int(round(x*60)))
flight

"""**Menampilkan statistik data**

---
Menampilkan jumlah data, rataan, standar deviasi, nilai minimum, quartil 1, quartil 2, quartil 3 dan nilai maksimum yang terdapat dalam dataset.

"""

data_table.enable_dataframe_formatter()
flight.describe()

"""**Menganalisis outliers pada variabel duration**

---
Mencari dan menghapus outlier pada variabel duration.

"""

# Mencari Outlier Pada Variabel duration
sns.boxplot(x=flight['duration'])

"""**Menganalisis outliers pada variabel days_left**

---
Mencari dan menghapus outlier pada variabel days_left.

"""

# Mencari Outlier pada variabel days_left
sns.boxplot(x=flight['days_left'])

"""**Mengeliminasi outliers dengan metode IQR**

---
Mengeliminasi outliers dengan Metode IQR, sehingga data outliers hilang.

"""

# Hapus Outlier dengan IQR
Q1 = flight.quantile(0.25)
Q3 = flight.quantile(0.75)

IQR = Q3-Q1
flight = flight[~((flight<(Q1-1.5*IQR)) | (flight>(Q3+1.5*IQR))).any(axis=1)]

flight.shape

"""**Encoder untuk kolom (variabel) stops**

---
Untuk memudahkan analisis data, variabel stops diubah dari kategorikal menjadi numerical
"""

# Label Encoder Untuk Variabel stops
def encoder(x):
  if(x=='zero'):
    return 0
  elif(x=='one'):
    return 1
  else:
    return 2

flight['stops'] = flight['stops'].apply(encoder)
flight[['stops']]

"""**Mengelompokkan Data Kategorikal dan Numerikal**

---
Mengelompokkan data ke dalam dua kategori, yaitu categorical_feature dan numerical_feature
"""

categorical_feature = ['airline', 'source_city', 'departure_time',
                       'arrival_time', 'destination_city', 'class']
numerical_feature = ['duration', 'days_left', 'price', 'stops']

"""**Visualisasi Variabel airline**

---
Memvisualisasikan sebaran data variabel airline dengan menggunakan diagram batang.

"""

feature = categorical_feature[0]
count = flight[feature].value_counts()
percent = 100*flight[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""**Visualisasi variabel source_city**

---
Memvisualisasikan sebaran data variabel source_city dengan menggunakan diagram batang.
"""

feature = categorical_feature[1]
count = flight[feature].value_counts()
percent = 100*flight[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""**Visualisasi variabel departure_time**

---
Memvisualisasikan sebaran data variabel departure_time dengan menggunakan diagram batang.

"""

feature = categorical_feature[2]
count = flight[feature].value_counts()
percent = 100*flight[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""**Visualisasi variabel arrival_time**

---
Memvisualisasikan sebaran data variabel arrival_time dengan menggunakan diagram batang.
"""

feature = categorical_feature[3]
count = flight[feature].value_counts()
percent = 100*flight[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""**Visualisasi variabel destination_city**

---
Memvisualisasikan sebaran data variabel destination_city dengan menggunakan diagram batang.

"""

feature = categorical_feature[4]
count = flight[feature].value_counts()
percent = 100*flight[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""**Visualisasi variabel class**

---
Memvisualisasikan sebaran data variabel clas dengan menggunakan diagram batang.
"""

feature = categorical_feature[5]
count = flight[feature].value_counts()
percent = 100*flight[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""**Membuat histogram untuk data numerik**

---
Membuat histogram untuk seluruh data numerik

"""

flight.hist(bins=50, figsize=(20,15))
plt.show()

"""**Visualisasi variabel secara multivariative**

---
Memvisualisasikan variabel dengan menggunakan analisis multivariative

"""

cat_features = flight.select_dtypes(include='object').columns.to_list()
 
for col in cat_features:
  sns.catplot(x=col, y="price", kind="bar", dodge=False, height = 4, aspect = 3,
              data=flight, palette="Set3")
  plt.title("Rata-rata 'price' Relatif terhadap - {}".format(col))

"""**Analisis Multivariative Pada Variabel Numerikal**

---
Analisis Multivariative Pada Variabel Numerikal
"""

sns.pairplot(flight, diag_kind = 'kde')

"""**Membuat correlation matrix**

---
Membuat correlation matrix untuk mengetahui korelasi data
"""

plt.figure(figsize=(10, 8))
correlation_matrix = flight.corr().round(2)
 
# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""**Data Preprocessing Lanjutan**

---
Melakukan data preprocessing lanjutan, dengan cara drop kolom/variabel days_left (karena tidak relevan)

"""

flight.drop(['days_left'], axis=1, inplace = True)

"""**Menampilkan update data**

---
Menampilkan update terbaru setelah dihilangkan variabel days_left

"""

flight

"""**Melakukan one hot encoding**

---
Melakukan one hot encoding pada seluruh data kategorikal
"""

from sklearn.preprocessing import  OneHotEncoder
flight = pd.concat([flight, pd.get_dummies(flight['airline'], prefix='airline')],axis=1)
flight = pd.concat([flight, pd.get_dummies(flight['departure_time'], prefix='departure_time')],axis=1)
flight = pd.concat([flight, pd.get_dummies(flight['arrival_time'], prefix='arrival_time')],axis=1)
flight = pd.concat([flight, pd.get_dummies(flight['class'], prefix='class')],axis=1)
flight = pd.concat([flight, pd.get_dummies(flight['source_city'], prefix='class')],axis=1)
flight = pd.concat([flight, pd.get_dummies(flight['destination_city'], prefix='class')],axis=1)
flight.drop(['airline','departure_time', 'arrival_time','class', 'source_city', 'destination_city'],
            axis=1, inplace=True)

"""**Cek data setelah one hot encoding**

---
Cek data setelah one hot encoding
"""

flight.info()

"""**Split Data**

---

Split data menjadi data latih (training data) dan data uji (test data)
"""

X = flight.drop(['price'],axis= 1)
y = flight['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

"""**Normalisasi data X_train**

---

Normalisasi data X_train
"""

from sklearn.preprocessing import StandardScaler
 
numerical_features = ['duration', 'stops']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

"""**Normalisasi data X_test**

---

Normalisasi data X_test
"""

scaler = StandardScaler()
scaler.fit(X_test[numerical_features])
X_test[numerical_features] = scaler.transform(X_test.loc[:, numerical_features])
X_test[numerical_features].head()

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""**Membuat dataframe untuk menampung nilai mean absolute error hasil pelatihan model**

---
Membuat dataframe untuk menampung nilai mean absolute error hasil pelatihan model
"""

# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mae', 'test_mae'], 
                      columns=['RandomForest', 'Boosting'])

"""**Melatih model dengan menggunakan Random Forest**

---
Melatih model dengan menggunakan Random Forest
"""

# Impor library yang dibutuhkan
from sklearn.ensemble import RandomForestRegressor
 
# buat model prediksi
RF = RandomForestRegressor(n_estimators=100, max_depth=64, max_features='sqrt', 
                           random_state=1, n_jobs=-1, verbose = 2, warm_start = True)
RF.fit(X_train, y_train)
 
models.loc['train_mae','RandomForest'] = mean_absolute_error(y_pred=RF.predict(X_train), y_true=y_train)

"""**Melatih model dengan menggunakan Boosting Algorithm**

---
Melatih model dengan menggunakan Boosting Algorithm
"""

from sklearn.ensemble import AdaBoostRegressor
 
boosting = AdaBoostRegressor(n_estimators = 100, learning_rate=1.0, random_state=1)                             
boosting.fit(X_train, y_train)
models.loc['train_mae','Boosting'] = mean_absolute_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""**Evaluasi Hasil training**

---
Melakukan evaluasi hasil training dengan melihat hasil Mean Absolute Error.

"""

# Buat variabel mae yang isinya adalah dataframe nilai mae data train dan test pada masing-masing algoritma
mae = pd.DataFrame(columns=['train', 'test'], index=['RF','Boosting'])
 
# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'RF': RF, 'Boosting': boosting}
 
# Hitung Mean Absolute Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mae.loc[name, 'train'] = mean_absolute_error(y_true=y_train, y_pred=model.predict(X_train))
    mae.loc[name, 'test'] = mean_absolute_error(y_true=y_test, y_pred=model.predict(X_test))
 
# Panggil mae
mae

"""**Visualisasi Mean Absolute Error**

---
Visualisasi Mean Absolute Error dengan menggunakan diagram
"""

fig, ax = plt.subplots()
mae.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

"""**Tes Nilai Hasil Prediksi Dibandingkan Nilai Sebenarnya**

---
Visualisasi Mean Absolute Error dengan menggunakan diagram
"""

prediksi = X_test.iloc[:10].copy()
pred_dict = {'y_true':y_test[:10]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)
 
pd.DataFrame(pred_dict)

"""**Menghitung 10% MAE**

---Menghitung 10% MAE untuk mengetahui keandalan model yang dibuat
"""

#Menghitung 10% Nilai Mean Absolute Error (MAE)
mae_target = (flight['price'].max() - flight['price'].min()) * 10/100
print(mae_target)

"""**Kesimpulan**
Nilai MAE yang dihasilkan oleh Random Forest pada saat training dan testing kurang dari 10%, sehingga model sudah dapat dikatakan baik (good fit).
"""