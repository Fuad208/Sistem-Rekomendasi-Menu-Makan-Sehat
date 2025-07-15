import streamlit as st
import pickle
import pandas as pd
from sklearn.cluster import KMeans

# CSS untuk styling
st.markdown(
    """
    <style>
    .stTitle {
        font-size: 36px;
        color: #FFA500;
        text-align: center;
    }
    .stHeader {
        font-size: 24px;
        color: #2E8B57;
    }
    .stSubheader {
        font-size: 20px;
        color: #4682B4;
        margin-top: 20px;
    }
    .stMarkdown {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

class FoodRecom:
    def show(self):
        st.title("🍽️ Rekomendasi Makanan 🍽️")

        # Load the saved model
        with open('CP_kmeans.pkl', 'rb') as f:
            model = pickle.load(f)

        # Baca dataset
        food_df = pd.read_csv('nutrition.csv')

        # Bagian Input Kebutuhan Harian
        st.header("Masukkan Kebutuhan Harian Anda")
        st.markdown('<p class="stMarkdown">Masukkan informasi kebutuhan harian Anda di bawah ini.</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            daily_calorie_needs = st.number_input("Kalori Harian:", min_value=1000, max_value=5000, step=100, value=1000)
        with col2:
            daily_proteins_needs = st.number_input("Protein Harian (gram):", min_value=10, max_value=300, step=10, value=10)
        with col3:
            daily_carbs_needs = st.number_input("Karbohidrat Harian (gram):", min_value=50, max_value=500, step=10, value=50)
        
        # Tambahkan pilihan jumlah makanan yang ingin direkomendasikan
        num_recommendations = st.number_input("Jumlah Rekomendasi Makanan:", min_value=1, max_value=10, step=1, value=1)

        if st.button("Generate Menu Makanan"):
            st.header("Menu Makanan Harian Anda")
            st.markdown('<p class="stMarkdown">Berikut adalah rekomendasi makanan berdasarkan kebutuhan harian Anda:</p>', unsafe_allow_html=True)

            # Fungsi untuk merekomendasikan makanan berdasarkan kebutuhan user
            def recommend_foods(model, food_df, calorie_needs, proteins_needs, carbs_needs, excluded_foods, num_recommendations):
                # Prediksi cluster untuk kebutuhan yang diberikan
                user_cluster = model.predict([[calorie_needs, proteins_needs, carbs_needs]])[0]
                
                # Dapatkan makanan dalam cluster yang sama
                food_df['cluster'] = model.predict(food_df[['calories', 'proteins', 'carbohydrate']])
                recommended_foods = food_df[(food_df['cluster'] == user_cluster) & (~food_df['name'].isin(excluded_foods))]
                
                # Filter makanan untuk mencocokkan kebutuhan kalori, protein, karbo dengan lebih dekat
                recommended_foods = recommended_foods.iloc[(recommended_foods[['calories', 'proteins', 'carbohydrate']] - [calorie_needs, proteins_needs, carbs_needs]).abs().sum(axis=1).argsort()[:num_recommendations]]
                
                return recommended_foods

            # Hitung kebutuhan untuk setiap kali makan
            breakfast_needs = (daily_calorie_needs * 0.25, daily_proteins_needs * 0.25, daily_carbs_needs * 0.25)
            lunch_needs = (daily_calorie_needs * 0.30, daily_proteins_needs * 0.30, daily_carbs_needs * 0.30)
            snack_needs = (daily_calorie_needs * 0.15, daily_proteins_needs * 0.15, daily_carbs_needs * 0.15)
            dinner_needs = (daily_calorie_needs * 0.30, daily_proteins_needs * 0.30, daily_carbs_needs * 0.30)

            # Melacak makanan yang sudah direkomendasikan
            excluded_foods = []

            # Rekomendasikan makanan untuk setiap kali makan
            breakfast_recommendations = recommend_foods(model, food_df, *breakfast_needs, excluded_foods, num_recommendations)
            excluded_foods.extend(breakfast_recommendations['name'].tolist())

            lunch_recommendations = recommend_foods(model, food_df, *lunch_needs, excluded_foods, num_recommendations)
            excluded_foods.extend(lunch_recommendations['name'].tolist())

            snack_recommendations = recommend_foods(model, food_df, *snack_needs, excluded_foods, num_recommendations)
            excluded_foods.extend(snack_recommendations['name'].tolist())

            dinner_recommendations = recommend_foods(model, food_df, *dinner_needs, excluded_foods, num_recommendations)
            excluded_foods.extend(dinner_recommendations['name'].tolist())

            # Fungsi untuk menampilkan rekomendasi makanan dalam bentuk tabel
            def display_recommendations(title, recommendations):
                st.subheader(title)
                st.dataframe(recommendations[['name', 'calories', 'proteins', 'carbohydrate']].reset_index(drop=True))

            # Tampilkan makanan yang direkomendasikan
            display_recommendations("🍳 Makanan yang Direkomendasikan untuk Sarapan:", breakfast_recommendations)
            display_recommendations("🍲 Makanan yang Direkomendasikan untuk Makan Siang:", lunch_recommendations)
            display_recommendations("🍎 Makanan yang Direkomendasikan untuk Snack:", snack_recommendations)
            display_recommendations("🍛 Makanan yang Direkomendasikan untuk Makan Malam:", dinner_recommendations)

            # Hitung total kalori, protein, dan karbohidrat dari rekomendasi makanan
            total_calories = sum(breakfast_recommendations['calories']) + sum(lunch_recommendations['calories']) + sum(snack_recommendations['calories']) + sum(dinner_recommendations['calories'])
            total_proteins = sum(breakfast_recommendations['proteins']) + sum(lunch_recommendations['proteins']) + sum(snack_recommendations['proteins']) + sum(dinner_recommendations['proteins'])
            total_carbs = sum(breakfast_recommendations['carbohydrate']) + sum(lunch_recommendations['carbohydrate']) + sum(snack_recommendations['carbohydrate']) + sum(dinner_recommendations['carbohydrate'])

            # Tampilkan hasil total kalori, protein, dan karbohidrat
            st.subheader("Estimasi Total Kebutuhan Harian yang Terpenuhi:")
            st.write(f"**Total Kalori:** {total_calories} kalori")
            st.write(f"**Total Protein:** {total_proteins} gram")
            st.write(f"**Total Karbohidrat:** {total_carbs} gram")