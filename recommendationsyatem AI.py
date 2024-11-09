import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Sample movie dataset
data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Memento'],
    'genre': ['Action, Sci-Fi', 'Action, Adventure, Sci-Fi', 'Adventure, Drama, Sci-Fi', 'Action, Crime, Drama', 'Mystery, Thriller'],
    'description': [
        'A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.',
        'A thief who enters the dreams of others to steal secrets from their subconscious is given the inverse task of planting an idea into the mind of a CEO.',
        'A team of explorers travel through a wormhole in space in an attempt to ensure humanity\'s survival.',
        'When the menace known as The Joker emerges from his mysterious past, he wreaks havoc and chaos on the people of Gotham.',
        'A man with short-term memory loss attempts to track down his wife\'s killer.'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)
# Create a TF-IDF vectorizer to process the movie descriptions
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the movie descriptions
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'])
# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Convert to a DataFrame for easier interpretation
cosine_sim_df = pd.DataFrame(cosine_sim, index=df['title'], columns=df['title'])
def recommend_movies(movie_title, cosine_sim_matrix, movie_df, num_recommendations=3):
    # Get the index of the movie that matches the title
    idx = movie_df[movie_df['title'] == movie_title].index[0]
    
    # Get the pairwise similarity scores of all movies with the selected movie
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the most similar movies
    sim_scores = sim_scores[1:num_recommendations+1]
    
    # Get movie indices and corresponding similarity scores
    movie_indices = [i[0] for i in sim_scores]
    movie_scores = [i[1] for i in sim_scores]
    
    # Return the titles of the recommended movies
    recommended_movies = movie_df['title'].iloc[movie_indices]
    return recommended_movies.tolist()

# Test the system
recommended_movies = recommend_movies('Inception', cosine_sim, df, num_recommendations=3)

# Print the recommended movies
print("Recommended Movies:")
for movie in recommended_movies:
    print(movie)
