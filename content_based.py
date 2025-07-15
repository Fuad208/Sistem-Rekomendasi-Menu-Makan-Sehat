import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentBased:
    def __init__(self, df_path='nutrition.csv'):
        self.df = pd.read_csv(df_path)
        self.features = ['calories', 'proteins', 'fat', 'carbohydrate']


    def recommend(self, selected_items, top_n=5):
        selected = self.df[self.df['name'].isin(selected_items)]
        vector = selected[self.features].mean().values.reshape(1, -1)
        similarities = cosine_similarity(self.df[self.features], vector)
        self.df['similarity'] = similarities
        return self.df.sort_values(by='similarity', ascending=False).head(top_n)
