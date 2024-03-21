import streamlit as st

def run_program():
    # Judul halaman
    st.set_page_config(
    page_title="Indo Stock",
    page_icon="🔫"
    )
    
    st.markdown("""<h1 style='text-align: center;'>My Stock Forecast</h1>""", unsafe_allow_html=True)
    # st.markdown("_Created by Kevin_", unsafe_allow_html=True)
    st.image("https://th.bing.com/th/id/OIG3.6QjRqg5YProOXqaHU0nW?w=1024&h=1024&rs=1&pid=ImgDetMain")
        
    st.write("""
    ## Pengantar  
    Dalam beberapa tahun terakhir, penggunaan AI untuk memprediksi harga saham telah menjadi topik yang menarik.
    Jenis jaringan saraf LSTM yang efektif untuk memahami dan memprediksi data berurutan, seperti data historis harga saham, salah satunya adalah LSTM.
    Oleh karena itu, kami menggunakan Implementasi LSTM pada aplikasi kami yang berjudul My Stock Forecast.
    """)
    
    st.write("""
    ## METODE Kerja MODEL LSTM
    Konsep dari jaringan saraf LSTM (Long Short-Term Memory) ialah sejenis desain jaringan saraf tiruan yang memodelkan dan memproses rangkaian data termasuk teks, suara, dan deret waktu.
    Dibandingkan dengan jaringan saraf rekursif (RNN) yang lebih sederhana, LSTM memiliki fitur sel memori yang memungkinkannya menyimpan informasi jangka panjang, menghindari masalah gradien hilang, dan mengontrol aliran informasi melalui gerbang lupa, gerbang masukan, dan gerbang keluaran.
    
    Hal ini membuat LSTM berguna dalam berbagai aplikasi, termasuk pemrosesan bahasa alami, pengenalan suara, terjemahan mesin, dan pemodelan deret waktu.
    Dengan teknik ini, kecerdasan buatan dapat belajar dari pola masa lalu untuk memprediksi pergerakan harga masa depan.
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
    - **Pembuatan Model LSTM**: Membangun arsitektur jaringan saraf LSTM menggunakan library seperti TensorFlow.
    - **Pelatihan Model**: Melatih model LSTM menggunakan data historis yang telah dipersiapkan sebelumnya.
    - **Evaluasi dan Prediksi**: Menggunakan model yang dilatih untuk melakukan prediksi dan mengevaluasi kinerjanya.
    """)
    
    
    st.header('Fitur aplikasi My Stock Forecast')
    st.write("""
    Pada aplikasi saya memiliki beberapa opsi pada side bar diantaranya:
    - Halaman Welcome
    - Halaman License pembuat
    - Halaman Prediksi             
             """)
    
    st.write("""
    ## Penutup
    Sekian Project 

    """)
    
if __name__ == '__main__':
    run_program()
