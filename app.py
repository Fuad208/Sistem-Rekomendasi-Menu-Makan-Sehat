import streamlit as st
import psycopg2
import re
import pandas as pd
import plotly.express as px
from db import get_connection
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_lottie import st_lottie
import json
import os

from kalkulator import KalkulatorKalori
from food_recom import FoodRecom
from content_based import ContentBased

# Koneksi ke database
conn = get_connection()
cursor = conn.cursor()

# Konfigurasi halaman
st.set_page_config(page_title="HealthyMe Apps", page_icon="🍽️")

# Background CSS
st.markdown("""
    <style>
     .stApp {
        background-image: url("https://img.freepik.com/free-vector/blurred-white-background-with-shine-effect_1017-33200.jpg");
        background-size: cover;
        transition: background 0.5s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Inisialisasi sesi pengguna
if 'user' not in st.session_state:
    st.session_state['user'] = None

# -------------------------------------------
# Fungsi Autentikasi
def signup(email, password, full_name):
    try:
        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            st.error("Masukkan email yang valid!")
            return
        if not password or len(password) < 6:
            st.error("Password minimal 6 karakter!")
            return

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        if cursor.fetchone():
            st.error("Email sudah terdaftar!")
            return

        cursor.execute(
            "INSERT INTO users (full_name, email, password) VALUES (%s, %s, %s)",
            (full_name, email, password)
        )
        conn.commit()
        st.success("Akun berhasil dibuat. Silakan login.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat signup: {str(e)}")

def login(email, password):
    try:
        cursor.execute("SELECT user_id, full_name, email FROM users WHERE email = %s AND password = %s", (email, password))
        result = cursor.fetchone()
        if result:
            st.session_state['user'] = {
                'user_id': result[0],
                'full_name': result[1],
                'email': result[2]
            }
            st.success("Berhasil login!")
            return True
        else:
            st.error("Email atau password salah.")
            return False
    except Exception as e:
        st.error(f"Kesalahan saat login: {str(e)}")
        return False

def logout():
    st.session_state['user'] = None
    st.success("Berhasil logout!")

# -------------------------------------------
# Fungsi Tampilan
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def show_login_signup():
    st.title("Selamat datang di HealtyME: Teman Sehat Anda!")
    st.write("Temukan kesehatan melalui pola makan yang tepat.")

    lottie_path = os.path.join('asset', 'konselor_food.json')
    if os.path.exists(lottie_path):
        lottie_animation = load_lottiefile(lottie_path)
        st_lottie(lottie_animation, height=300)

    with st.sidebar:
        st.header("Autentikasi")
        if st.session_state['user'] is None:
            choice = st.radio("Pilih opsi", ["Login", "Signup"])

            if choice == "Signup":
                st.subheader("Buat Akun Baru")
                email = st.text_input("Email", key="signup_email")
                password = st.text_input("Password", type="password", key="signup_password")
                full_name = st.text_input("Nama Lengkap", key="signup_full_name")
                if st.button("Daftar"):
                    signup(email, password, full_name)

            elif choice == "Login":
                st.subheader("Masuk ke Akun")
                email = st.text_input("Email", key="login_email")
                password = st.text_input("Password", type="password", key="login_password")
                if st.button("Masuk"):
                    if login(email, password):
                        st.rerun()

def sidebar_main_app():
    user_info = st.session_state['user']
    with st.sidebar:
        st.header(f"Selamat datang, {user_info['full_name']} 👋")
        selected = option_menu(None, ["Home", 'BMI Calculator', 'Rekomendasi Menu', 'Menu Pilihanmu'],
                               icons=['house', 'calculator', 'book', 'star'])
        if st.button("Logout"):
            logout()
            st.rerun()
        return selected

def show_main_app():
    selected = sidebar_main_app()

    if selected == "Home":
        st.title("Selamat Datang di HealthyMe")
        st.write("Mari peroleh informasi nutrisi dan rekomendasi makanan sehat.")

        file_path = "nutrition.csv"
        df = pd.read_csv(file_path)

        with st.container():
            st.subheader("Data Nutrisi Makanan Sehat")
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc='sum', editable=True)
            gb.configure_side_bar()
            gridOptions = gb.build()

            AgGrid(df, gridOptions=gridOptions, enable_enterprise_modules=True, update_mode='MODEL_CHANGED')

            st.subheader("Scatter Plot Nutrisi Makanan Sehat")
            fig_scatter1 = px.scatter(df, x='calories', y='proteins', color='name',
                                      title='Scatter Plot Kalori vs Protein')
            st.plotly_chart(fig_scatter1, use_container_width=True)

            fig_scatter2 = px.scatter(df, x='calories', y='fat', color='name',
                                      title='Scatter Plot Kalori vs Lemak')
            st.plotly_chart(fig_scatter2, use_container_width=True)

    elif selected == "BMI Calculator":
        KalkulatorKalori().show()

    elif selected == "Rekomendasi Menu":
        FoodRecom().show()

    elif selected == "Menu Pilihanmu":
        st.header("🍽️ Rekomendasi Berdasarkan Makanan Favorit")
        cb = ContentBased()

        try:
            all_items = cb.df['name'].tolist()
            selected_items = st.multiselect("Pilih makanan favorit kamu:", all_items)

            if selected_items:
                results = cb.recommend(selected_items)

                st.write("Hasil rekomendasi:", results.head())

                if not results.empty:
                    st.subheader("Rekomendasi makanan serupa:")
                    cols_to_show = [col for col in ['name', 'calories', 'proteins', 'fat', 'carbohydrate'] if col in results.columns]
                    st.dataframe(results[cols_to_show])
                else:
                    st.warning("Tidak ada rekomendasi yang bisa ditampilkan.")
            else:
                st.info("Silakan pilih minimal satu makanan.")
        except Exception as e:
            st.error(f"Terjadi error saat menampilkan rekomendasi: {str(e)}")



# -------------------------------------------
# Jalankan Aplikasi
if st.session_state['user'] is None:
    show_login_signup()
else:
    show_main_app()
