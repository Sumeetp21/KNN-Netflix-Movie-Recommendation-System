import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ==============================
# Step 1: Load Dataset
# ==============================

ratings = pd.read_csv("ml-latest-small/ratings.csv")
movies = pd.read_csv("ml-latest-small/movies.csv")


print("Ratings Dataset:")
print(ratings.head())

print("\nMovies Dataset:")
print(movies.head())


# ==============================
# Step 2: Create User-Movie Matrix
# ==============================

movie_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)


# Replace missing ratings with 0

movie_matrix.fillna(0, inplace=True)


print("\nMovie Matrix Shape:")
print(movie_matrix.shape)



# ==============================
# Step 3: Train KNN Model
# ==============================


model = NearestNeighbors(
    metric='cosine',
    algorithm='brute'
)


# Transpose because we need movie similarity

model.fit(movie_matrix.T)



# ==============================
# Step 4: Recommendation Function
# ==============================


def recommend_movie(movie_name, number_of_recommendations=5):

    # Search movie name

    movie_search = movies[
        movies['title'].str.contains(
            movie_name,
            case=False,
            regex=False
        )
    ]


    if movie_search.empty:
        print("Movie not found")
        return


    # Get movie ID

    movie_id = movie_search.iloc[0]['movieId']


    # Find similar movies

    distances, indices = model.kneighbors(
        movie_matrix.T.loc[[movie_id]],
        n_neighbors=number_of_recommendations + 1
    )


    print("\nMovie Selected:")
    print(movie_search.iloc[0]['title'])


    print("\nRecommended Movies:\n")


    for i in range(1, len(indices.flatten())):

        movie_index = indices.flatten()[i]

        distance = distances.flatten()[i]


        recommended_movie_id = movie_matrix.T.index[movie_index]


        movie_title = movies[
            movies.movieId == recommended_movie_id
        ]['title'].values[0]


        similarity = round((1-distance)*100,2)


        print(
            f"{i}. {movie_title}  --> Similarity: {similarity}%"
        )



# ==============================
# Step 5: Test Recommendation
# ==============================


recommend_movie("Toy Story")