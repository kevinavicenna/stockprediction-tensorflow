import streamlit as st


def run_program():
    # Data identitas diri (contoh data tetap)
    data_identitas = {
    "Nama": "Kevin Avicenna Widiarto",
    "Umur": 22,
    "Jenis Kelamin": "Laki-laki",
    "Alamat": "Jalan Contoh No. 123"
    }
    # Judul halaman
    st.set_page_config(
    page_title="Indo Stock",
    page_icon="🔫"
    )

    st.title("Stock Forecast for Market Indonesia")
    st.subheader("Made by Kevin Avicenna")
    # st.image("https://tradebrains.in/features/wp-content/uploads/2021/07/stock-market-news-trade-brains.jpg")

    # Sidebar
    st.sidebar.success("select menu above")

    
    st.write("""
    ## Pengantar
    Penggunaan kecerdasan buatan (AI) untuk memprediksi pergerakan harga saham telah menjadi topik yang menarik
    dalam beberapa tahun terakhir. Salah satu teknik yang populer adalah menggunakan jaringan saraf LSTM (Long
    Short-Term Memory) untuk memodelkan pola dan tren dalam data saham.

    LSTM adalah jenis jaringan saraf yang efektif dalam memahami dan memprediksi data berurutan, seperti data
    historis harga saham. Dengan menggunakan teknik ini, AI dapat belajar dari pola masa lalu untuk memprediksi
    pergerakan harga masa depan.
    """)

    st.write("""
    ## Bagaimana LSTM Bekerja pada Prediksi Saham?
    LSTM memiliki kemampuan untuk mempertahankan dan menggunakan informasi jangka panjang dari data historis.
    Dalam konteks prediksi saham, LSTM dapat mempelajari hubungan kompleks antara variabel-variabel seperti
    harga saham sebelumnya, volume perdagangan, dan faktor-faktor lainnya.

    LSTM melakukan ini dengan mempertimbangkan informasi sebelumnya dalam jangka waktu tertentu, dan kemudian
    memutuskan bagaimana informasi tersebut akan mempengaruhi pergerakan harga saham pada waktu selanjutnya.
    """)

    st.write("""
    ## Implementasi LSTM untuk Prediksi Saham
    Implementasi LSTM untuk prediksi saham melibatkan beberapa langkah:
    
    - **Persiapan Data**: Pengumpulan dan persiapan data historis saham yang akan digunakan untuk melatih model.
    - **Pembuatan Model LSTM**: Membangun arsitektur jaringan saraf LSTM menggunakan library seperti TensorFlow atau PyTorch.
    - **Pelatihan Model**: Melatih model LSTM menggunakan data historis yang telah dipersiapkan sebelumnya.
    - **Evaluasi dan Prediksi**: Menggunakan model yang dilatih untuk melakukan prediksi dan mengevaluasi kinerjanya.
    """)

    st.write("""
    ## Penutup
    Penggunaan LSTM dalam pengembangan AI untuk prediksi saham merupakan bidang yang terus berkembang. Namun,
    penting untuk diingat bahwa prediksi pasar keuangan tetaplah sulit dan tidak ada jaminan keakuratannya.
    Kesalahan dalam prediksi harga saham dapat terjadi karena faktor-faktor yang sulit diprediksi oleh model.
    
    Kombinasi pengetahuan pasar, analisis fundamental, dan penggunaan AI dapat membantu para investor dalam
    pengambilan keputusan, namun demikian, tetap diperlukan kewaspadaan.
    """)
    # Menampilkan informasi identitas diri
    st.write(f"Nama: {data_identitas['Nama']}")
    st.write(f"Umur: {data_identitas['Umur']} tahun")
    st.write(f"Jenis Kelamin: {data_identitas['Jenis Kelamin']}")
    st.write(f"Alamat: {data_identitas['Alamat']}")

# # Menampilkan footer
# st.footer("Ini adalah contoh halaman identitas diri.")
    
if __name__ == '__main__':
    run_program()

# def main():
#     st.title('Guide Aplikasi Streamlit')
    
#     st.header('Pendahuluan')
#     st.write("""
#     Stock Prediction adalah sebuah framework open-source yang memungkinkan Anda membuat aplikasi web dengan cepat
#     menggunakan Python. Dengan Streamlit, Anda dapat membuat aplikasi web untuk visualisasi data, machine
#     learning, analisis data, dan banyak lagi dengan mudah.
#     """)
    
#     st.header('Langkah Pertama: Instalasi')
#     st.write("""
#     Untuk menggunakan Streamlit, Anda perlu menginstalnya terlebih dahulu. Anda dapat menginstalnya melalui
#     pip (package manager Python) dengan menjalankan perintah berikut di terminal atau command prompt:
    
#     ```
#     pip install streamlit
#     ```
    
#     Setelah terinstal, Anda dapat membuat aplikasi Streamlit menggunakan editor kode favorit Anda.
#     """)
    
#     st.header('Membuat Aplikasi Pertama')
#     st.write("""
#     Untuk membuat aplikasi pertama, buatlah file Python baru dan impor library Streamlit. Kemudian, gunakan
#     fungsi-fungsi Streamlit seperti `st.title()`, `st.write()`, `st.header()`, dan lainnya untuk membuat
#     tampilan aplikasi Anda.
    
#     Contoh sederhana aplikasi Streamlit:
    
#     ```python
#     import streamlit as st
    
#     def main():
#         st.title('Aplikasi Sederhana')
#         st.write('Halo, ini adalah aplikasi sederhana menggunakan Streamlit.')
    
#     if __name__ == '__main__':
#         main()
#     ```
    
#     Simpan file tersebut dan jalankan dengan perintah `streamlit run nama_file.py` pada terminal.
#     """)
    
#     st.header('Fitur-Fitur Streamlit')
#     st.write("""
#     Streamlit memiliki beragam fitur yang memudahkan pembuatan aplikasi web:
    
#     - **Widget Interaktif**: Seperti `st.button()`, `st.slider()`, `st.selectbox()` untuk interaksi pengguna.
#     - **Visualisasi**: Dukungan untuk menampilkan plot dengan mudah seperti `st.line_chart()`, `st.bar_chart()`.
#     - **Tata Letak**: Membuat tata letak dengan mudah menggunakan `st.columns()`, `st.expander()`.
#     - **Kustomisasi**: Memungkinkan kustomisasi dengan CSS dan HTML melalui `st.markdown()` dan `st.write()`.
#     """)
    
#     st.header('Referensi dan Dokumentasi')
#     st.write("""
#     Untuk informasi lebih lanjut dan dokumentasi lengkap, kunjungi situs resmi Streamlit di [streamlit.io](https://streamlit.io).
    
#     Anda juga dapat membaca dokumentasi resmi di [dokumentasi Streamlit](https://docs.streamlit.io).
#     """)
    
# if __name__ == '__main__':
#     main()
