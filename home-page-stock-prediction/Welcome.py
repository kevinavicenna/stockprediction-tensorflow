import streamlit as st

def run_program():
    st.set_page_config(
        page_title="My Stock Forecast",
        page_icon="💹"
    )
    
    st.markdown("""<h1 style='text-align: center;'>My Stock Forecast 💹</h1>""", unsafe_allow_html=True)
    st.image("https://th.bing.com/th/id/OIG3.6QjRqg5YProOXqaHU0nW?w=1024&h=1024&rs=1&pid=ImgDetMain")
        
    st.write("""
    ## Arsitecture LSTM (Long Short Term Memory)
    Konsep dari Neural Network LSTM (Long Short-Term Memory) ialah sejenis desain jaringan saraf tiruan yang memodelkan dan memproses rangkaian data termasuk teks, suara, dan deret waktu.
    Dibandingkan dengan Recursive Neural Network (RNN) yang lebih sederhana, LSTM memiliki fitur sel memori yang memungkinkannya menyimpan informasi jangka panjang, menghindari masalah gradien hilang, dan mengontrol aliran informasi melalui gerbang lupa, gerbang masukan, dan gerbang keluaran.
    
    Hal ini membuat LSTM berguna dalam berbagai aplikasi, termasuk nlp, pengenalan suara, terjemahan, dan time series.
    Dengan teknik ini, kecerdasan buatan dapat belajar dari pola masa lalu untuk memprediksi pergerakan harga masa depan.
    """)

    st.write("""
    ## Bagaimana LSTM Bekerja pada Prediksi Saham?
    LSTM memiliki kemampuan untuk mempertahankan dan menggunakan informasi jangka panjang dari data historis.
    Dalam konteks prediksi saham, LSTM dapat mempelajari hubungan kompleks antara variabel-variabel seperti
    harga saham sebelumnya, volume perdagangan, dan faktor-faktor lainnya.

    Namun,LSTM melakukan ini dengan mempertimbangkan informasi sebelumnya dalam jangka waktu tertentu, dan kemudian
    memutuskan bagaimana informasi tersebut akan mempengaruhi pergerakan harga saham pada waktu selanjutnya.
    """)
        
    st.header('Pengenalan Aplikasi My Stock Forecast')
    st.write("""
    Pada aplikasi saya memiliki beberapa opsi pada side bar diantaranya:
    - Halaman Welcome
    - Halaman Dashboard
    - Halaman License 
     
    Pada halaman dashboard menampilkan:
    - Rentang data stock yang dipilih
    - Grafik pergerakan harga
    - Fitur Seasonal Decompose yang terdiri dari Trend, Seasonality dan Residual
    - Model Prediksi LSTM 
    
    Note: 
    Kekurangan aplikasi hanya ditraining menggunakan satu data stock 
    Jika menggunakannya untuk stock lain kemungkinan besar hasilnya jauh berbeda
    Maka harus dilakukan training data ulang dengan data yang dituju
             """)

    
if __name__ == '__main__':
    run_program()
