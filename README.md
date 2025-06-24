# Laporan Proyek Machine Learning - Rifal Ariya Yusuftrian

## Domain Proyek

Saya memilih proyek prediksi harga saham PT Unilever Indonesia Tbk (UNVR.JK) menggunakan metode Long Short-Term Memory (LSTM), yang merupakan bagian dari deep learning. Permasalahan ini sangat relevan karena prediksi harga saham dapat digunakan oleh investor maupun pelaku pasar modal untuk menentukan strategi investasi yang lebih tepat.

Menurut hasil riset dari Guresen et al., 2011, LSTM terbukti efektif dalam menangani data runtun waktu seperti harga saham karena kemampuannya mengingat informasi historis dalam jangka panjang.

## Business Understanding

### Problem Statements

- Bagaimana memprediksi harga saham harian UNVR di masa depan berdasarkan data historis?
- Bagaimana membangun model deep learning (LSTM) yang mampu mempelajari pola harga saham?

### Goals

- Menghasilkan prediksi harga saham UNVR 10 hari ke depan.
- Membangun model LSTM dengan akurasi tinggi berdasarkan data historis dari tahun 2010.

### Solution Statement

- Saya menggunakan model LSTM untuk menangani data time series karena sifat arsitekturnya yang cocok dalam memahami pola historis.
- Model diimprove melalui pemilihan parameter seperti jumlah neuron LSTM dan jumlah epoch.

## Data Understanding

Dataset yang digunakan adalah data harga saham harian UNVR dari Yahoo Finance. Dataset ini mencakup periode dari 1 Januari 2010 hingga 24 Juni 2025.

🔗 [Yahoo Finance - UNVR.JK](https://finance.yahoo.com/quote/UNVR.JK/history)

### Fitur pada dataset:

- `Open`: Harga saat pembukaan
- `High`: Harga tertinggi harian
- `Low`: Harga terendah harian
- `Close`: Harga penutupan
- `Adj Close`: Harga yang disesuaikan
- `Volume`: Jumlah saham yang diperdagangkan

### Kondisi Dataset
Dataset yang digunakan memiliki jumlah total 3.900 baris data dan 7 kolom (fitur). Periode data mencakup 1 Januari 2010 hingga 24 Juni 2025.

Berikut informasi kualitas data:

- Missing Values (Null):
Hasil pemeriksaan menggunakan data.isnull().sum() menunjukkan bahwa tidak terdapat missing values pada semua kolom, sehingga tidak diperlukan proses imputasi data.

- Duplikasi:
Pemeriksaan menggunakan data.duplicated().sum() menunjukkan bahwa tidak ditemukan baris data yang duplikat.

### Visualisasi Tren Harga Saham UNVR

```python
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Harga Penutupan (Close)', color='blue')
plt.title('Tren Harga Penutupan Saham UNVR')
plt.xlabel('Tanggal')
plt.ylabel('Harga (IDR)')
plt.grid(True)
plt.legend()
plt.show()
```
<img width="611" alt="image" src="https://github.com/user-attachments/assets/80394c65-6117-4338-9a12-de2345044ab5" />


## Data Preparation

### Seleksi Kolom Target

Saya memilih kolom `Close` sebagai target prediksi. Kolom lain seperti Open, High, Low dan Volume tidak digunakan dalam model ini.

### Normalisasi Data

Data dinormalisasi dengan `MinMaxScaler` agar berada dalam rentang 0-1.

```python
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)
```

### Membentuk Data Time Series

Saya menggunakan sliding window 60 hari untuk membentuk data input untuk model LSTM.

```python
n_past = 60
X, y = [], []
for i in range(n_past, len(scaled_data)):
    X.append(scaled_data[i - n_past:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)
```

## Modeling

### Model LSTM

LSTM (Long Short-Term Memory) adalah salah satu jenis Recurrent Neural Network (RNN) yang dirancang untuk memproses data runtun waktu. LSTM memiliki kemampuan untuk mengingat informasi jangka panjang melalui struktur khusus yang disebut cell state, forget gate, input gate, dan output gate.

Dalam konteks proyek ini, LSTM mampu mempelajari pola historis dari harga saham yang bersifat time series. Misalnya, jika harga saham biasanya naik setelah penurunan selama 3 hari, LSTM dapat belajar dari pola tersebut dan menggunakannya untuk membuat prediksi masa depan.

Model dibangun menggunakan:

- 2 lapisan LSTM dengan 64 dan 32 unit neuron.
- 1 lapisan Dense output dengan 1 neuron untuk menghasilkan nilai prediksi harga saham.
- Optimizer: adam, karena cepat dan stabil dalam konvergensi.
- Loss function: mean_squared_error yang umum untuk regresi.

Pemilihan LSTM didasarkan pada kecocokan arsitekturnya dengan data sekuensial seperti harga saham, serta hasil riset sebelumnya yang menunjukkan keunggulan LSTM dalam memprediksi data keuangan.

```python
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)
```

---

## Prediction 🔮
Setelah model LSTM berhasil dilatih, selanjutnya melakukan prediksi harga saham untuk 10 hari ke depan. Proses ini tidak hanya memprediksi satu langkah ke depan, tetapi menggunakan pendekatan recursive forecasting, yaitu memprediksi hari pertama, lalu menggunakan prediksi tersebut sebagai input untuk memprediksi hari kedua, dan seterusnya hingga hari ke-10.

```python
last_60_days = scaled_data[-60:]
future_input = last_60_days.values.reshape(1, 60, 1)

future_predictions = []
for _ in range(10):
    pred = model.predict(future_input)[0]
    future_predictions.append(pred)
    future_input = np.append(future_input[:, 1:, :], [[pred]], axis=1)

future_predictions = scaler.inverse_transform(future_predictions)
```

## Visualisai Hasil Prediksi
Selanjutnya adalah memvisualisasikan harga historis 100 hari terakhir bersamaan dengan hasil prediksi 10 hari ke depan. Garis putus-putus berwarna oranye menunjukkan tren hasil prediksi dari model LSTM.

```python
# Membuat range tanggal untuk 10 hari ke depan
future_dates = pd.date_range(start=close_prices.index[-1] + pd.Timedelta(days=1), periods=10)

# Gabungkan harga terakhir + hasil prediksi
combined_prices = [close_prices.iloc[-1]['Close']] + future_predictions.flatten().tolist()
combined_dates = [close_prices.index[-1]] + list(future_dates)

combined_df = pd.DataFrame({'Price': combined_prices}, index=combined_dates)

# Visualisasikan
plt.figure(figsize=(12,6))
plt.plot(close_prices[-100:], label='Historis', color='blue')
plt.plot(combined_df, label='Prediksi 10 Hari', color='orange', linestyle='--')
plt.title('Prediksi Harga Saham UNVR 10 Hari ke Depan')
plt.xlabel('Tanggal')
plt.ylabel('Harga (IDR)')
plt.legend()
plt.grid(True)
plt.show()
```
<img width="609" alt="image" src="https://github.com/user-attachments/assets/4c17f1fe-f2d7-47fa-8530-596084760c77" />



## Evaluation

### Metrik Evaluasi

Model dievaluasi menggunakan MSE, RMSE, MAE, dan MAPE:

```python
mse = mean_squared_error(actual_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actual_prices, predicted_prices)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
```

### Hasil Evaluasi

| Metrik | Nilai    | Keterangan                                               |
|--------|----------|----------------------------------------------------------|
| MSE    | 15450.06 | Kesalahan kuadrat rata-rata cukup rendah                 |
| RMSE   | 124.30   | Rata-rata kesalahan prediksi dalam satuan harga          |
| MAE    | 88.44    | Selisih rata-rata absolut antara prediksi dan aktual     |
| MAPE   | 3.37%    | Model memiliki tingkat akurasi tinggi dengan error kecil |

---

Interpretasi Hasil Evaluasi dalam Konteks Business Understanding
Hasil evaluasi model menunjukkan nilai error yang rendah:
- MAPE sebesar 3.37%, yang artinya rata-rata kesalahan prediksi hanya sekitar 3% dari nilai aktual.
- RMSE dan MAE masing-masing sebesar 124.30 dan 88.44, masih dalam batas toleransi pergerakan harga saham harian UNVR.

Kaitan dengan Problem Statements dan Goals:

- ✔️ Problem Statement 1 (memperkirakan harga harian UNVR):
Model berhasil memprediksi harga saham dengan akurasi tinggi.
- ✔️ Problem Statement 2 (membangun model yang mempelajari pola harga):
LSTM mampu mengenali pola jangka pendek dan menengah dari harga saham sebelumnya, menunjukkan bahwa model mampu belajar pola dari data historis.
- ✔️ Goals (prediksi 10 hari ke depan):
Model digunakan untuk membuat prediksi multi-step ke depan, dan hasilnya menunjukkan tren yang selaras dengan harga aktual.

Dampak terhadap Business Goals:
- Investor dapat menggunakan prediksi ini sebagai salah satu referensi dalam pengambilan keputusan beli/jual.
- Prediksi jangka pendek ini juga dapat membantu manajemen risiko investasi, terutama dalam strategi harian atau mingguan.

### Visualisasi Prediksi vs Aktual

```python
plt.figure(figsize=(12,6))
plt.plot(test_data.index[n_past:], actual_prices, label='Aktual', color='blue')
plt.plot(test_data.index[n_past:], predicted_prices, label='Prediksi', color='orange', linestyle='--')
plt.title('Perbandingan Harga Aktual vs Prediksi pada Data Uji')
plt.xlabel('Tanggal')
plt.ylabel('Harga (IDR)')
plt.legend()
plt.grid(True)
plt.show()
```
<img width="608" alt="image" src="https://github.com/user-attachments/assets/2818907c-1eea-426c-aa70-9ce1f2d01077" />



## Kesimpulan

Proyek ini berhasil membangun model prediksi harga saham PT Unilever Indonesia Tbk (UNVR.JK) menggunakan metode Long Short-Term Memory (LSTM), yang merupakan bagian dari deep learning. Berdasarkan evaluasi menggunakan metrik MSE, RMSE, MAE, dan MAPE, model menunjukkan performa yang cukup baik dengan tingkat akurasi tinggi (MAPE sebesar 3.37%).

Dengan menggunakan data historis sejak tahun 2010 dan pendekatan time series menggunakan window 60 hari, model mampu mempelajari pola pergerakan harga saham dan memberikan estimasi harga 10 hari ke depan secara relatif akurat. Hal ini membuktikan bahwa metode LSTM efektif untuk digunakan dalam pemodelan data runtun waktu di sektor keuangan.

