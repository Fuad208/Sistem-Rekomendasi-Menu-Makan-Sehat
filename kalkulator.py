import streamlit as st
from PIL import Image
import os
import time

class KalkulatorKalori:
    def show(self):
        # Fungsi untuk menghitung BMI
        def calculate_bmi(weight, height):
            height_m = height / 100
            return weight / (height_m ** 2)
        
        # Fungsi untuk menentukan kategori BMI
        def determine_bmi_category(bmi):
            if bmi < 18.5:
                return "Kurus (Kurang Berat Badan)😫👎"
            elif 18.5 <= bmi < 24.9:
                return "Normal (Berat Badan Ideal)😆👍"
            elif 25 <= bmi < 29.9:
                return "Berlebih Berat Badan🤔"
            elif 30 <= bmi < 34.9:
                return "Obesitas Tingkat 1🙁"
            elif 35 <= bmi < 39.9:
                return "Obesitas Tingkat 2 (Berisiko Tinggi)😣"
            else:
                return "Obesitas Tingkat 3 (Sangat Berisiko)😨"

        # Fungsi untuk menghitung berat badan ideal menggunakan Metode Devine
        def calculate_ideal_weight_devine(gender, height):
            height_in = height / 2.54
            if gender == 'Pria':
                return 50 + 2.3 * (height_in - 60)
            else:
                return 45.5 + 2.3 * (height_in - 60)

        # Fungsi untuk menghitung BMR menggunakan berat badan ideal
        def calculate_bmr_mifflin(gender, weight, height, age):
            if gender == 'Pria':
                return (10 * weight) + (6.25 * height) - (5 * age) + 5
            else:
                return (10 * weight) + (6.25 * height) - (5 * age) - 161

        # Fungsi untuk menghitung TDEE berdasarkan BMR dan tingkat aktivitas
        def calculate_tdee(bmr, activity_level):
            if activity_level == 'Sangat jarang : sangat jarang olahraga':
                return bmr * 1.2
            elif activity_level == 'Jarang : jarang olahraga (1-3 hari/minggu)':
                return bmr * 1.375
            elif activity_level == 'Normal : normal olahraga (3-5 hari/minggu)':
                return bmr * 1.55
            elif activity_level == 'Sering : sering olahraga (6-7 hari/minggu)':
                return bmr * 1.725
            else:
                return bmr * 1.9
            
        # Fungsi untuk menentukan distribusi makronutrien
        def calculate_macros(tdee, goal):
            if goal == 'Menurunkan Berat Badan':
                tdee -= 500
            elif goal == 'Menambah Berat Badan':
                tdee += 500

            carbs = 0.5 * tdee / 4  # 4 kalori per gram karbohidrat, 0.5  ratio makronutrien
            protein = 0.3 * tdee / 4  # 4 kalori per gram protein, 0.3  ratio makronutrien
            fat = 0.2 * tdee / 9  # 9 kalori per gram lemak, 0.2  ratio makronutrien

            return tdee, carbs, protein, fat
        
        # Menampilkan judul dan penjelasan alat
        st.markdown('<div style="text-align: center;font-size:40px;font-weight:bold;color:black;">Kalkulator Kebutuhan Kalori dan Makronutrien Harian</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center;font-size:18px;">Alat ini membantu Anda menghitung kebutuhan kalori harian dan distribusi makronutrien berdasarkan data pribadi dan tujuan Anda.</p>', unsafe_allow_html=True)

        # Menampilkan gambar
        image_path = os.path.join('asset', 'hitung_kalori.jpg')
        image = Image.open(image_path)
        st.image(image, use_container_width
=True)

        # Menampilkan panduan penggunaan
        st.markdown('<h2 style="text-align: center; color: black;">Masukkan Data Anda</h2>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center;">Masukkan data Anda pada kolom yang tersedia, lalu klik tombol "Hitung" untuk melihat hasil perhitungan.</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"], key="gender")
            weight = st.number_input("Berat Badan (kg) 🏋️", min_value=0.0, format="%.2f", key="weight")
            height = st.number_input("Tinggi Badan (cm) 📏", min_value=0.0, format="%.2f", key="height")            
        with col2:
            age = st.number_input("Usia (tahun)", min_value=0, format="%d", key="age")
            activity_level = st.selectbox("Tingkat Aktivitas 🏃‍♂️",
                                          ["Sangat jarang : sangat jarang olahraga", 
                                           "Jarang : jarang olahraga (1-3 hari/minggu)", 
                                           "Normal : normal olahraga  (3-5 hari/minggu)", 
                                           "Sering : sering olahraga (6-7 hari/minggu)", 
                                           "Sangat sering : sangat sering olahraga (setiap hari bisa 2x dalam sehari/pekerjaan fisik)"], key="activity_level")
            goal = st.selectbox("Tujuan 🎯", ["Menurunkan Berat Badan", 
                                              "Mempertahankan Berat Badan", 
                                              "Menambah Berat Badan"], key="goal")

        hitung_button = st.button("Hitung")

        if hitung_button:
            # Memeriksa apakah masukan berat badan, tinggi badan, dan usia sudah diisi dengan benar
            if weight <= 0 or height <= 0 or age <= 0:
                st.warning("⚠️ Pastikan untuk mengisi berat badan, tinggi badan, dan usia dengan benar!")
            else:
                # Menampilkan spinner loading
                with st.spinner("Sedang menghitung..."):
                    # Simulasi proses perhitungan dengan menunggu 2 detik
                    time.sleep(1.5)
                
                # Hitung BMI
                bmi = calculate_bmi(weight, height)
                bmi_category = determine_bmi_category(bmi)

                # Hitung berat badan ideal menggunakan Metode Devine
                ideal_weight = calculate_ideal_weight_devine(gender, height)
                
                # Hitung BMR menggunakan berat badan ideal
                bmr = calculate_bmr_mifflin(gender, ideal_weight, height, age)
                
                # Hitung TDEE berdasarkan BMR dan tingkat aktivitas
                tdee = calculate_tdee(bmr, activity_level)
                
                # Hitung distribusi makronutrien dan kalori harian
                adjusted_tdee, carbs, protein, fat = calculate_macros(tdee, goal)
                
                # Menampilkan hasil perhitungan
                st.subheader("Hasil Perhitungan Nutrisi Harian")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="BMI 📊", value=f"{bmi:.2f}", help="BMI adalah indikator yang digunakan untuk mengukur status gizi seseorang berdasarkan berat dan tinggi badan. Semakin rendah nilai BMI, semakin rendah berat badan relatif terhadap tinggi badan. Kategori BMI umumnya dapat didefinisikan sebagai berikut:\n\n- Kurus (Kurang Berat Badan): BMI kurang dari 18.5\n- Normal (Berat Badan Ideal): BMI antara 18.5 dan 24.9\n- Berlebih Berat Badan: BMI antara 25 dan 29.9\n- Obesitas Tingkat 1: BMI antara 30 dan 34.9\n- Obesitas Tingkat 2 (Berisiko Tinggi): BMI antara 35 dan 39.9\n- Obesitas Tingkat 3 (Sangat Berisiko): BMI 40 atau lebih")
                    st.markdown(f"**Kategori BMI**: {bmi_category}")
                    st.metric(label="Berat Badan Ideal ⚖️", value=f"{ideal_weight:.2f} kg", help="Berat badan ideal dihitung menggunakan Metode Devine berdasarkan tinggi badan dan jenis kelamin.")
                    st.metric(label="BMR 🔥", value=f"{bmr:.2f} kalori/hari", help="Basal Metabolic Rate (BMR) adalah jumlah kalori yang dibutuhkan tubuh untuk mempertahankan fungsi dasar kehidupan saat istirahat.")

                with col2:
                    st.metric(label="TDEE 🏃‍♀️", value=f"{tdee:.2f} kalori/hari",  help="Total Daily Energy Expenditure (TDEE) adalah jumlah total kalori yang dibutuhkan setiap hari berdasarkan tingkat aktivitas. Ini mencakup semua aktivitas fisik dan metabolisme basal Anda.")
                    st.metric(label="Kalori Harian 🍽️", value=f"{adjusted_tdee:.2f} kalori/hari", help="Jumlah kalori harian disesuaikan berdasarkan tujuan Anda (menurunkan, mempertahankan, atau menambah berat badan).")
                    st.metric(label="Karbohidrat 🍞", value=f"{carbs:.2f} gram/hari")
                    st.metric(label="Protein 🥩", value=f"{protein:.2f} gram/hari")
                    st.metric(label="Lemak 🥑", value=f"{fat:.2f} gram/hari")

                # Button untuk clear data
                clear_button_placeholder = st.empty()
                with clear_button_placeholder:
                    if st.button("Hitung ulang"):
                        st.experimental_rerun()


        # Penjelasan lebih lanjut tentang hasil perhitungan
        with st.expander("**Tips Kesehatan**"):
            st.markdown("""
                - **Karbohidrat, Protein, dan Lemak**: Distribusi makronutrien yang disesuaikan dengan kebutuhan kalori harian, dihitung berdasarkan rasio standar.
                - **Makanlah Makanan Sehat**: Prioritaskan makanan alami seperti buah, sayuran, biji-bijian utuh, dan protein berkualitas.
                - **Berolahraga secara Teratur**: Lakukan olahraga secara teratur sesuai dengan kemampuan Anda untuk menjaga kebugaran tubuh dan kesehatan jantung.
                - **Tetap Terhidrasi**: Minumlah air yang cukup setiap hari untuk menjaga keseimbangan cairan dalam tubuh.
            """)