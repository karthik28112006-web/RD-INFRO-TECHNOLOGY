# app.py
import streamlit as st
from utils import (
    load_data,
    precompute_similarity,
    get_recommendations,
    get_genre_recommendations,
)

st.set_page_config(page_title="Movie Recommender", layout="centered")

for key in ["show_matcher", "show_genre"]:
    if key not in st.session_state:
        st.session_state[key] = False

def reset_matcher():
    st.session_state.show_matcher = False

def reset_genre():
    st.session_state.show_genre = False

@st.cache_data
def get_cached_data():
    df = load_data()
    sim_matrix = precompute_similarity(df)
    return df, sim_matrix

try:
    df, sim_matrix = get_cached_data()
except requests.exceptions.RequestException as e:
    st.error(f"Network error while fetching movie data: {e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

st.title("Movie Recommendation System")

tab1, tab2 = st.tabs(["🔍 Movie Matcher", "🎭 Genre Explorer"])

with tab1:
    selected_title = st.selectbox(
        "Select a movie:",
        sorted(df["title"].unique()),
        key="matcher_title",
        on_change=reset_matcher,
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Get Recommendations", key="btn_matcher"):
            st.session_state.show_matcher = True
    with col2:
        if st.button("Clear Results", key="clear_matcher"):
            st.session_state.show_matcher = False

    if st.session_state.show_matcher:
        results = get_recommendations(selected_title, df, sim_matrix, top_n=5)
        if results:
            st.subheader("Top 5 Recommendations")
            for i, r in enumerate(results, 1):
                st.write(f"{i}. {r}")
        else:
            st.warning("No recommendations found.")

with tab2:
    genre_list = sorted(set(g for genres in df["genres"] for g in genres.split()))
    selected_genre = st.selectbox(
        "Select a genre:",
        genre_list,
        key="genre_choice",
        on_change=reset_genre,
    )
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Get Genre Picks", key="btn_genre"):
            st.session_state.show_genre = True
    with col4:
        if st.button("Clear Results", key="clear_genre"):
            st.session_state.show_genre = False

    if st.session_state.show_genre:
        results = get_genre_recommendations(selected_genre, df, top_n=5)
        if results:
            st.subheader("Top 5 Highly-Rated")
            for i, r in enumerate(results, 1):
                st.write(f"{i}. {r}")
        else:
            st.warning("No matches found.")