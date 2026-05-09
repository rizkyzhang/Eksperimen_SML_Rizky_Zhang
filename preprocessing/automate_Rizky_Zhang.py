import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Load raw data
df = pd.read_csv("../sample_superstore_raw.csv", encoding="latin-1")
print(f"Initial shape and size: {df.shape}, {df.size}")
print(f"\nInitial dtypes:\n{df.dtypes}")

# 2. Missing values
print("\nMissing values:")
print(df.isnull().sum())
df = df.dropna()

# 3. Menghapus duplikat
print("\nDuplicate rows:")
print(df.duplicated().sum())
df = df.drop_duplicates()

# 4. Menghapus kolom yang tidak relevan
# Kolom-kolom yang dihapus tidak bisa digunakan dalam prediksi Sales, terutama kolom high cardinality seperti 'Product Name' saat dilakukan one hot encoding akan menghasilkan banyak noise.
df = df.drop(columns=['Row ID', 'Order ID', 'Customer ID', 'Customer Name',
                       'Product ID', 'Product Name', 'Ship Date', 'Country', 'City',
                       'State', 'Postal Code', 'YearMonth'], errors='ignore')

# 5. Deteksi outlier dan penanganan
# Karena Sales merupakan variabel target, nilai outlier dalam sales dapat mengakibatkan overfit saat training.
Q1 = df['Sales'].quantile(0.25)
Q3 = df['Sales'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Sales'] >= Q1 - 1.5 * IQR) & (df['Sales'] <= Q3 + 1.5 * IQR)]

# 6. Encode data kategorikal
# ML training akan menggunakan algoritma Linear Regression dan Random Forest, algoritma tersebut tidak memproses data string seperti kategori 'Furniture' sehingga harus dikonversi ke nilai 0/1 dalam kolom baru.
df = pd.get_dummies(df, columns=['Ship Mode', 'Segment', 'Region', 'Category', 'Sub-Category'])

# 7. Normalisasi data numerik
scaler = MinMaxScaler()
df[['Quantity', 'Discount', 'Profit']] = scaler.fit_transform(df[['Quantity', 'Discount', 'Profit']])

# 8. Binning kolom Discount (Low / Medium / High)
bins = [0, 0.2, 0.4, 1.0]
labels = ['Low', 'Medium', 'High']
df['Discount_Bin'] = pd.cut(df['Discount'], bins=bins, labels=labels, include_lowest=True)
print(f"\nFinal shape and size after preprocessing: {df.shape}, {df.size}")
print(f"\nFinal dtypes after preprocessing:\n{df.dtypes}")

# 9. Save preprocessed csv
df.to_csv("sample_superstore_preprocessing.csv", index=False)
print(f"\nSaved preprocessed.csv")
