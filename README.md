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
