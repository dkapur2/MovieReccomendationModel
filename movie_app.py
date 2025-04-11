import pandas as pd
import numpy as np
import re
import gdown
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title("🎬 Movie Recommendation App")

# Load data
@st.cache_data
def load_data():
    # Download the plain CSV from Google Drive
    gdown.download(
        "https://drive.google.com/uc?id=1MtdfWJLmwPEwABPR5MRMjVbHmOojeIBd",
        "ratings_10percent.csv",
        quiet=False
    )

    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings_10percent.csv")

    return movies, ratings

movies, ratings = load_data()

# Clean titles
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

movies["clean_title"] = movies["title"].apply(clean_title)

# Vectorize movie titles
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

# Search based on input
def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

# Recommend similar movies
def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
    
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

# Streamlit UI
movie_title = st.text_input("Enter a movie title:", "Toy Story")

if len(movie_title) > 5:
    results = search(movie_title)
    if not results.empty:
        selected_movie = results.iloc[0]
        st.markdown(f"### Top 10 recommendations for **{selected_movie['title']}**:")
        recommendations = find_similar_movies(selected_movie["movieId"])
        st.table(recommendations)
    else:
        st.warning("No results found. Try another title.")
