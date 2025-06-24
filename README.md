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

Dataset yang digunakan adalah data harga saham harian UNVR dari Yahoo Finance. Dataset ini mencakup periode dari 1 Januari 2010 hingga 24 Juni 2025, dan memiliki lebih dari 500 baris data.

🔗 [Yahoo Finance - UNVR.JK](https://finance.yahoo.com/quote/UNVR.JK/history)

### Fitur pada dataset:

- `Open`: Harga saat pembukaan
- `High`: Harga tertinggi harian
- `Low`: Harga terendah harian
- `Close`: Harga penutupan
- `Adj Close`: Harga yang disesuaikan
- `Volume`: Jumlah saham yang diperdagangkan

### Visualisasi Tren Harga Penutupan

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

Model dibangun menggunakan 2 lapisan LSTM (64 dan 32 unit) dan 1 Dense layer.

```python
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=20, batch_size=32)
```

Model ini dipilih karena keunggulannya dalam mempelajari pola sekuensial data historis.

---

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

## Kesimpulan

Proyek ini berhasil membangun model prediksi harga saham PT Unilever Indonesia Tbk (UNVR.JK) menggunakan metode Long Short-Term Memory (LSTM), yang merupakan bagian dari deep learning. Berdasarkan evaluasi menggunakan metrik MSE, RMSE, MAE, dan MAPE, model menunjukkan performa yang cukup baik dengan tingkat akurasi tinggi (MAPE sebesar 3.37%).

Dengan menggunakan data historis sejak tahun 2010 dan pendekatan time series menggunakan window 60 hari, model mampu mempelajari pola pergerakan harga saham dan memberikan estimasi harga 10 hari ke depan secara relatif akurat. Hal ini membuktikan bahwa metode LSTM efektif untuk digunakan dalam pemodelan data runtun waktu di sektor keuangan.

Ke depan, model dapat dikembangkan lebih lanjut dengan:
- Menambahkan fitur eksternal seperti indikator teknikal (MACD, RSI, dll),
- Menyesuaikan arsitektur model (misalnya menggunakan GRU atau Attention-based models),
- Menambahkan validasi silang dan fine-tuning hyperparameter lebih optimal.

Proyek ini menjadi langkah awal yang solid dalam menerapkan machine learning untuk pengambilan keputusan investasi berbasis data.

