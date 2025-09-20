import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# -----------------------------
# Session state init
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "login"  # can be 'login' or 'recommender'

# -----------------------------
# Login & Register Page
# -----------------------------
def login_register_page():
    st.title("📚 Books Recommender System - Login/Register")
    menu = ["Login", "Register"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Register
    if choice == "Register":
        st.subheader("📝 Register")
        name = st.text_input("Full Name")
        college = st.text_input("College Name")
        domain = st.text_input("Domain / Field of Interest")
        roll_no = st.text_input("Roll Number")
        contact = st.text_input("Contact Number")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")

        if st.button("Register"):
            if all([name, college, domain, roll_no, contact, gender, username, password]):
                user_data = pd.DataFrame([[name, college, domain, roll_no, contact, gender, username, password]],
                                         columns=["Name", "College", "Domain", "RollNo", "Contact", "Gender", "Username", "Password"])
                if os.path.exists("users.csv"):
                    user_data.to_csv("users.csv", mode="a", header=False, index=False)
                else:
                    user_data.to_csv("users.csv", index=False)
                st.success("Registration successful! You can now login.")
            else:
                st.error("Please fill all fields.")

    # Login
    elif choice == "Login":
        st.subheader("🔑 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if os.path.exists("users.csv"):
                users = pd.read_csv("users.csv")
                if any((users["Username"] == username) & (users["Password"] == password)):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.page = "recommender"
                else:
                    st.error("Invalid credentials")
            else:
                st.error("No users registered yet.")

# -----------------------------
# Recommender Page
# -----------------------------
def recommender_page():
    st.sidebar.success(f"✅ Logged in as {st.session_state.username}")
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.page = "login"
        st.experimental_rerun()  # only here we can safely rerun

    st.subheader(f"Welcome, {st.session_state.username}!")

    # Load datasets
    books = pd.read_csv('data/BX-Books.csv', sep=";", on_bad_lines='skip', encoding='latin-1')
    ratings = pd.read_csv('data/BX-Book-Ratings.csv', sep=";", on_bad_lines='skip', encoding='latin-1')

    # Reduce dataset
    top_books = ratings['ISBN'].value_counts().head(5000).index
    top_users = ratings['User-ID'].value_counts().head(5000).index
    ratings_small = ratings[ratings['ISBN'].isin(top_books) & ratings['User-ID'].isin(top_users)]

    book_pivot = ratings_small.pivot(index='ISBN', columns='User-ID', values='Book-Rating').fillna(0)
    sparse_matrix = csr_matrix(book_pivot.values)

    model = NearestNeighbors(algorithm='brute', metric='cosine')
    model.fit(sparse_matrix)

    isbn_to_title = dict(zip(books['ISBN'], books['Book-Title']))
    title_to_isbn = {v:k for k,v in isbn_to_title.items()}

    # Fetch poster online via Google Books API
    @st.cache_data
    def fetch_poster_online(title):
        query = title.replace(" ", "+")
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}"
        try:
            response = requests.get(url)
            data = response.json()
            thumbnail = data['items'][0]['volumeInfo']['imageLinks']['thumbnail']
            if thumbnail.startswith("http:"):
                thumbnail = thumbnail.replace("http:", "https:")
            return thumbnail
        except:
            return None

    # Recommend book function
    def recommend_book(book_title):
        if book_title not in title_to_isbn:
            return [], []
        isbn = title_to_isbn[book_title]
        if isbn not in book_pivot.index:
            return [], []
        book_id_array = np.where(book_pivot.index == isbn)[0]
        if len(book_id_array) == 0:
            return [], []
        book_id = book_id_array[0]
        distances, suggestion = model.kneighbors(
            book_pivot.iloc[book_id,:].values.reshape(1,-1),
            n_neighbors=6
        )
        suggested_isbns = book_pivot.index[suggestion[0]].tolist()
        suggested_titles = [isbn_to_title[i] for i in suggested_isbns]
        posters = [fetch_poster_online(isbn_to_title[i]) for i in suggested_isbns]
        return suggested_titles, posters

    # Book selection
    book_names = books[books['ISBN'].isin(book_pivot.index)]['Book-Title'].dropna().unique().tolist()
    selected_book = st.selectbox("Type or select a book", book_names)

    # Display selected book cover
    if selected_book:
        poster = fetch_poster_online(selected_book)
        if poster:
            st.image(poster, width=120)

    # Show recommendations
    if selected_book and st.button('Show Recommendation'):
        recommended_books, poster_url = recommend_book(selected_book)
        if not recommended_books:
            st.info("No recommendations available.")
        else:
            cols = st.columns(5)
            for i, col in enumerate(cols):
                if i < len(recommended_books):
                    col.text(recommended_books[i])
                    if poster_url[i]:
                        col.image(poster_url[i], width=100)
                    # Amazon & Flipkart links
                    amazon_link = f"https://www.amazon.in/s?k={recommended_books[i].replace(' ', '+')}"
                    flipkart_link = f"https://www.flipkart.com/search?q={recommended_books[i].replace(' ', '+')}"
                    col.markdown(f"[Buy on Amazon]({amazon_link})")
                    col.markdown(f"[Buy on Flipkart]({flipkart_link})")

    # Compact rating & review
    st.subheader("⭐ Rate & Review")
    col1, col2 = st.columns([1,3])
    with col1:
        rating = st.slider("⭐ Rating", 1, 5, 3)
    with col2:
        review = st.text_area("📝 Write a review", height=80)

    if st.button("Submit Review"):
        st.success(f"Thank you for your rating of {rating} and review!")
        review_data = pd.DataFrame([[st.session_state.username, selected_book, rating, review]],
                                   columns=["User", "Book", "Rating", "Review"])
        if os.path.exists("reviews.csv"):
            review_data.to_csv("reviews.csv", mode="a", header=False, index=False)
        else:
            review_data.to_csv("reviews.csv", index=False)

# -----------------------------
# MAIN APP
# -----------------------------
if st.session_state.page == "login" or not st.session_state.logged_in:
    login_register_page()
elif st.session_state.page == "recommender" and st.session_state.logged_in:
    recommender_page()
