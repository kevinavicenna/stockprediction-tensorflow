import streamlit as st

def run_program():

    st.set_page_config(
    page_title="Indo Stock",
    page_icon="🔫"
    )

    st.markdown("""<h1 style='text-align: center;'>License</h1>""", unsafe_allow_html=True)
    # st.image("https://th.bing.com/th/id/OIG3.6QjRqg5YProOXqaHU0nW?w=1024&h=1024&rs=1&pid=ImgDetMain")
    
    st.write("""
    MIT License

    Copyright (c) 2023 Kevin Avicenna Widiarto

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """)
    
if __name__ == '__main__':
    run_program()