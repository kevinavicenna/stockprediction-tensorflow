import streamlit as st

def main():
    st.title('Guide Aplikasi Streamlit')
    
    st.header('Pendahuluan')
    st.write("""
    Stock Prediction adalah sebuah framework open-source yang memungkinkan Anda membuat aplikasi web dengan cepat
    menggunakan Python. Dengan Streamlit, Anda dapat membuat aplikasi web untuk visualisasi data, machine
    learning, analisis data, dan banyak lagi dengan mudah.
    """)
    
    st.header('Langkah Pertama: Instalasi')
    st.write("""
    Untuk menggunakan Streamlit, Anda perlu menginstalnya terlebih dahulu. Anda dapat menginstalnya melalui
    pip (package manager Python) dengan menjalankan perintah berikut di terminal atau command prompt:
    
    ```
    pip install streamlit
    ```
    
    Setelah terinstal, Anda dapat membuat aplikasi Streamlit menggunakan editor kode favorit Anda.
    """)
    
    st.header('Membuat Aplikasi Pertama')
    st.write("""
    Untuk membuat aplikasi pertama, buatlah file Python baru dan impor library Streamlit. Kemudian, gunakan
    fungsi-fungsi Streamlit seperti `st.title()`, `st.write()`, `st.header()`, dan lainnya untuk membuat
    tampilan aplikasi Anda.
    
    Contoh sederhana aplikasi Streamlit:
    
    ```python
    import streamlit as st
    
    def main():
        st.title('Aplikasi Sederhana')
        st.write('Halo, ini adalah aplikasi sederhana menggunakan Streamlit.')
    
    if __name__ == '__main__':
        main()
    ```
    
    Simpan file tersebut dan jalankan dengan perintah `streamlit run nama_file.py` pada terminal.
    """)
    
    st.header('Fitur-Fitur Streamlit')
    st.write("""
    Streamlit memiliki beragam fitur yang memudahkan pembuatan aplikasi web:
    
    - **Widget Interaktif**: Seperti `st.button()`, `st.slider()`, `st.selectbox()` untuk interaksi pengguna.
    - **Visualisasi**: Dukungan untuk menampilkan plot dengan mudah seperti `st.line_chart()`, `st.bar_chart()`.
    - **Tata Letak**: Membuat tata letak dengan mudah menggunakan `st.columns()`, `st.expander()`.
    - **Kustomisasi**: Memungkinkan kustomisasi dengan CSS dan HTML melalui `st.markdown()` dan `st.write()`.
    """)
    
    st.header('Referensi dan Dokumentasi')
    st.write("""
    Untuk informasi lebih lanjut dan dokumentasi lengkap, kunjungi situs resmi Streamlit di [streamlit.io](https://streamlit.io).
    
    Anda juga dapat membaca dokumentasi resmi di [dokumentasi Streamlit](https://docs.streamlit.io).
    """)
    
if __name__ == '__main__':
    main()
