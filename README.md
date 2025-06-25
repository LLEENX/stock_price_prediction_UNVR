# Laporan Proyek Machine Learning - Rifal Ariya Yusuftrian

## Domain Proyek

Saya memilih proyek prediksi harga saham PT Unilever Indonesia Tbk (UNVR.JK) menggunakan metode Long Short-Term Memory (LSTM), yang merupakan bagian dari deep learning. Permasalahan ini sangat relevan karena prediksi harga saham dapat digunakan oleh investor maupun pelaku pasar modal untuk menentukan strategi investasi yang lebih tepat.

Menurut hasil riset dari Guresen et al., 2011, LSTM terbukti efektif dalam menangani data runtun waktu seperti harga saham karena kemampuannya mengingat informasi historis dalam jangka panjang.

## Business Understanding

### Problem Statements

- Bagaimana cara memprediksi tren harga saham PT Unilever Indonesia Tbk (UNVR.JK) yang sangat fluktuatif, dengan mempertimbangkan pola historis yang kompleks dan bersifat non-linear?
- Apakah model LSTM mampu menghasilkan prediksi jangka pendek yang cukup akurat untuk membantu investor retail dalam mengambil keputusan beli/jual, dibandingkan pendekatan tradisional seperti moving average?

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
Dataset yang digunakan memiliki jumlah total sekitar 3.900 baris data dan 7 kolom (fitur). Periode data mencakup 1 Januari 2010 hingga 24 Juni 2025.

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

Model LSTM (Long Short-Term Memory)
LSTM (Long Short-Term Memory) adalah salah satu jenis Recurrent Neural Network (RNN) yang dirancang khusus untuk menangani data runtun waktu (time series) dengan kemampuan mengingat pola jangka panjang. Berbeda dari RNN biasa yang mudah mengalami vanishing gradient, LSTM memiliki struktur internal berupa "gate" yang memungkinkan kontrol atas informasi apa yang disimpan, dibuang, atau diteruskan ke langkah berikutnya.

## 🔍 Cara Kerja Intuitif LSTM

Setiap unit LSTM memiliki tiga komponen utama yang disebut *gate*, yaitu:

### 🔸 Forget Gate (fₜ)
Menentukan informasi apa yang perlu dilupakan dari memori jangka panjang (*cell state*) sebelumnya.
Misalnya, jika tren lama tidak lagi relevan untuk prediksi ke depan, gate ini akan menurunkannya.

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

- \( f_t \): nilai antara 0 dan 1 yang menentukan seberapa besar informasi sebelumnya dilupakan.
- \( h_{t-1} \): output dari langkah sebelumnya.
- \( x_t \): input saat ini.
- \( W_f \): bobot forget gate.
- \( b_f \): bias forget gate.
- \( \sigma \): fungsi aktivasi sigmoid.

---

### 🔸 Input Gate (iₜ) + Candidate Value (ĉₜ)
Mengontrol informasi baru apa yang akan ditambahkan ke memori.
*Candidate value* (ĉₜ) adalah informasi baru yang dihasilkan dari input saat ini dan akan disaring oleh *input gate* sebelum ditambahkan ke *cell state*.

  $$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) \\
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
$$

- \( i_t \): seberapa besar informasi baru yang disimpan.
- \( \tilde{C}_t \): kandidat informasi baru yang ingin ditambahkan ke *cell state*.
- \( W_i, W_C \): bobot input gate dan candidate.
- \( b_i, b_C \): bias masing-masing.

---

### Memperbarui Cell State (Cₜ)

Menggabungkan informasi dari *forget gate* dan *input gate*:

$$
C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t
$$

- \( C_t \): *cell state* saat ini.
- \( C_{t-1} \): *cell state* sebelumnya.

---

### 🔸 Output Gate (oₜ)
Menentukan output dan informasi ke langkah selanjutnya:

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \\
h_t = o_t \cdot \tanh(C_t)
$$

- \( o_t \): gate yang mengontrol output saat ini.
- \( h_t \): *hidden state* atau output untuk timestep ini.

> 💡 Mekanisme ini membuat LSTM mirip seperti otak kecil yang bisa memilih apa yang perlu diingat dan dilupakan, tergantung pada konteks saat itu.

---


## 🔁 Contoh Kasus Time Series

Dalam proyek **prediksi harga saham**:

- LSTM akan melihat tren harga saham selama **60 hari terakhir**,
- Menentukan pola historis penting (misalnya tren naik setelah koreksi 3 hari),
- Mengingat pola tersebut menggunakan *cell state*,
- Lalu memanfaatkannya untuk memprediksi harga ke depan.

---

## 🏗️ Arsitektur Model

Model dibangun menggunakan:

- **Dua lapisan LSTM:**
  - Lapisan pertama: 64 unit, dengan `return_sequences=True` agar mengirim seluruh urutan ke lapisan berikutnya.
  - Lapisan kedua: 32 unit.
- **Satu lapisan Dense** sebagai output dengan 1 neuron untuk menghasilkan prediksi harga akhir.
- **Optimizer**: `adam`, karena efisien dan cepat dalam konvergensi.
- **Loss Function**: `mean_squared_error`, cocok untuk regresi nilai numerik.

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

### ✅ Kaitan dengan Problem Statements dan Goals
- Problem Statement 1 (Memprediksi harga harian UNVR):
Model LSTM yang dibangun berhasil memprediksi harga saham dengan akurasi tinggi, dibuktikan oleh nilai MAPE sebesar 3.37%, yang termasuk dalam kategori error rendah untuk kasus prediksi pasar saham.

- Problem Statement 2 (Membangun model yang mampu mempelajari pola harga):
LSTM mampu menangkap pola historis dan tren jangka pendek hingga menengah dalam data harga saham, karena memori jangka panjangnya secara struktural cocok dengan data time series seperti ini.

- Goals (Prediksi 10 hari ke depan):
Model berhasil melakukan prediksi multi-step (10 hari ke depan) menggunakan teknik recursive forecasting, dengan hasil prediksi yang konsisten terhadap tren historis.

### 💡 Dampak terhadap Business Goals
- Model ini memberikan alat bantu kuantitatif bagi investor dalam menganalisis arah pergerakan harga saham jangka pendek, sehingga dapat mendukung pengambilan keputusan beli/jual yang lebih terinformasi.
- Dalam konteks manajemen risiko investasi, prediksi harga 10 hari ke depan juga dapat digunakan untuk menghindari keputusan spekulatif, khususnya bagi investor ritel yang lebih rentan terhadap volatilitas pasar.
- Keberhasilan model dalam memprediksi tren harga memberikan nilai praktis bagi pengembangan sistem pendukung keputusan (Decision Support System) di sektor keuangan.

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

