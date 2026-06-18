import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def do_preprocessing(input_path, output_path):
    print(f"[*] Membaca data mentah dari: {input_path}")
    
    # Validasi keberadaan berkas input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Data mentah tidak ditemukan di: {input_path}")
        
    # Membaca data menggunakan pandas
    df = pd.read_csv(input_path)
    
    # Membersihkan spasi pada nama kolom bawaan dataset jika ada
    df.columns = df.columns.str.strip()
    
    # Menangani Missing Values secara aman jika di masa depan ada data kosong masuk
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    
    # Menghapus kolom 'loan_id' karena tidak memiliki nilai prediktif
    if 'loan_id' in df.columns:
        df = df.drop(columns=['loan_id'])
        print("[+] Kolom 'loan_id' berhasil dihapus.")
        
    # Menghapus data duplikat (jika ada)
    df = df.drop_duplicates()
    
    # Handling data teks / kategorikal (education & self_employed) dengan LabelEncoder
    le = LabelEncoder()
    if 'education' in df.columns:
        df['education'] = le.fit_transform(df['education'].astype(str).str.strip())
    if 'self_employed' in df.columns:
        df['self_employed'] = le.fit_transform(df['self_employed'].astype(str).str.strip())
        
    # Mengubah target 'loan_status' menjadi biner (1: Approved, 0: Rejected)
    target_col = 'loan_status'
    if target_col in df.columns:
        df[target_col] = df[target_col].astype(str).str.strip().str.lower()
        df[target_col] = df[target_col].apply(lambda x: 1 if x == 'approved' else 0)
    print("[+] Encoding data kategorikal dan target 'loan_status' selesai.")
    
    # Standarisasi fitur numerik menggunakan StandardScaler
    scaler = StandardScaler()
    # Ambil semua kolom kecuali target 'loan_status'
    fitur_numerik = [col for col in df.columns if col != target_col]
    
    if fitur_numerik:
        df[fitur_numerik] = scaler.fit_transform(df[fitur_numerik])
        print("[+] Standarisasi skala fitur numerik selesai.")
        
    # Memastikan direktori folder tujuan sudah terbentuk otomatis
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Menyimpan file csv hasil akhir preprocessing
    df.to_csv(output_path, index=False)
    print(f"[+] Sukses! Berkas data bersih berhasil diekspor ke: {output_path}")
    
    return df

if __name__ == "__main__":
    # Tentukan base directory secara dinamis agar script dapat dijalankan dari folder mana saja
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path input
    INPUT_DATA = os.path.join(BASE_DIR, "loan_approval_raw", "dataset_raw.csv")
    
    # Path output
    OUTPUT_DATA = os.path.join(BASE_DIR, "preprocessing", "loan_approval_preprocessing", "dataset_clean.csv")
    
    print("\n" + "="*50)
    print("  MEMULAI PROSES OTOMATISASI PREPROCESSING DATA CLI  ")
    print("="*50)
    
    try:
        do_preprocessing(INPUT_DATA, OUTPUT_DATA)
        print("="*50)
        print("[INFO] Status Eksekusi: BERHASIL (100% Aman)")
        print("="*50 + "\n")
    except Exception as e:
        print(f"[ERROR] Terjadi kegagalan proses otomatisasi: {e}")
        print("="*50 + "\n")
        sys.exit(1)