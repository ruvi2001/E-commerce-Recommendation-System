import numpy as np
from flask import Flask, request, render_template, session, redirect, url_for
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
import random

app = Flask(__name__)

#load files
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

#database configuration
engine = create_engine('mysql+mysqldb://root:@localhost/ecom')
app.secret_key = "Secret Key"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql://root:1234@localhost/ecom"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


#define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)
    preferred_categories = db.Column(db.String(255), nullable=True)


#define your model class for the 'signin' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)



# for model based collaborative filtering
counts = train_data['ID'].value_counts()
train_data_final = train_data[train_data['ID'].isin(counts[counts >= 5].index)]
train_data_final = train_data_final.groupby(['ID', 'ProdID']).agg({'Rating': 'mean'}).reset_index()
#creating the interaction matrix of products and users based on ratings and replacing Nan
final_ratings_matrix = train_data_final.pivot(index = 'ID', columns = 'ProdID', values = 'Rating').fillna(0)

final_ratings_sparse = csr_matrix(final_ratings_matrix.values)
#Singular Value Decomposition
U, s, Vt = svds(final_ratings_sparse, k = 50) # here k is the number of latent features
sigma = np.diag(s)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
preds_train_data = pd.DataFrame(abs(all_user_predicted_ratings), columns = final_ratings_matrix.columns)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
# Construct diagonal array in SVD
sigma = np.diag(s)

preds_matrix = csr_matrix(preds_train_data.values)


#recommendation functions
#function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


def get_products_by_categories(train_data,preferred_categories):
    # Expand Category column by splitting on commas
    data_expanded = train_data.assign(Category=train_data['Category'].str.split(',')).explode('Category')  #The explode function then transforms these lists into individual rows

    # Strip any whitespace around the categories
    data_expanded['Category'] = data_expanded['Category'].str.strip()


    print("Filtering categories:", preferred_categories)
    print("Available categories in data:", data_expanded['Category'].unique())

    # Group by product and aggregate all categories into a set
    product_categories = data_expanded.groupby('Name')['Category'].apply(set)

    # Filter for products that contain all preferred categories
    # products_with_all_categories = product_categories[
    #     product_categories.apply(lambda x: set(preferred_categories).issubset(x))]

    # Filter the main data frame to get product details of the filtered products
    #filtered_products = data_expanded[data_expanded['Name'].isin(products_with_all_categories.index)]

    # Filter based on the preferred categories
    filtered_products = data_expanded[data_expanded['Category'].isin(preferred_categories)]

    #print("Filtered Products DataFrame:")
    #print(filtered_products)

    return filtered_products[['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']].drop_duplicates().head(8)


def content_based_recommendations(train_data, item_name, top_n=10):
    train_data['Tags'] = train_data['Tags'].fillna('')
    # Check if the item name exists in the training data
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = train_data[train_data['Name'] == item_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n + 1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = train_data.iloc[recommended_item_indices][
        ['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details




def rating_based_recommendations(train_data, top_n=8):
    # Calculate the average ratings for each product
    average_ratings = train_data.groupby(['Name', 'ReviewCount', 'Brand', 'ImageURL'])['Rating'].mean().reset_index()

    # Sort by rating in descending order to get top-rated items
    top_rated_items = average_ratings.sort_values(by='Rating', ascending=False)

    # Get the top N rated items
    rating_based_rec = top_rated_items.head(top_n)

    # Convert ratings and review count to integers
    rating_based_rec['Rating'] = rating_based_rec['Rating'].astype(int)
    rating_based_rec['ReviewCount'] = rating_based_rec['ReviewCount'].astype(int)

    return rating_based_rec[['Name', 'Rating', 'ReviewCount', 'Brand', 'ImageURL']]


def recommend_items(train_data,user_id, interactions_matrix, preds_matrix, num_recommendations=5):   #collaborative filtering model based
    # Print type of user_id for debugging
    print(f"Type of user_id: {type(user_id)}")

    # Ensure user_id is an integer
    if isinstance(user_id, csr_matrix):
        print("user_id is a csr_matrix. This is incorrect.")
        return None  # Handle this scenario or raise an error

    try:
        user_id = int(user_id)  # Ensure it's converted to int
    except ValueError as ve:
        print(f"Error converting user_id to int: {ve}")
        return None
    print(f" user id: {user_id}")
    # Ensure user ID is in the interaction matrix
    if user_id >= interactions_matrix.shape[0] or interactions_matrix.getrow(user_id).getnnz() == 0:
        print(f"Train user ID {user_id} not in interactions matrix")

    # Get the user's ratings from the actual and predicted interaction matrices
    user_ratings = interactions_matrix.getrow(user_id).toarray().reshape(-1)
    user_predictions = preds_matrix.getrow(user_id).toarray().reshape(-1)

    # Creating a dataframe with actual and predicted ratings columns
    temp = pd.DataFrame({'user_ratings': user_ratings, 'user_predictions': user_predictions})
    temp['Recommended Products'] = np.arange(len(user_ratings))
    temp = temp.set_index('Recommended Products')

    # Filtering the dataframe where actual ratings are 0, which implies that the user has not interacted with that product
    temp = temp.loc[temp.user_ratings == 0]

    # Recommending products with top predicted ratings
    temp = temp.sort_values('user_predictions', ascending=False)
    recommended_indices = temp.head(num_recommendations).index.tolist()
    # recommended_products = temp['user_predictions'].head(num_recommendations).index.tolist()

    # recommended_products = pd.DataFrame(recommended_products)
    recommended_products = train_data.loc[recommended_indices, ['Name', 'Rating', 'ReviewCount', 'Brand', 'ImageURL']]
    return recommended_products



#routes
#list of predefined image urls

random_image_urls = [
    "static/img_1.png",
    "static/img_2.png",
    "static/img_3.png",
    "static/img_4.png",
    "static/img_5.png",
    "static/img_6.png",
    "static/img_7.png",
    "static/img_8.png",
]


@app.route("/")
def index():

    #create a list of random image urls for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 150, 70, 100, 200, 106, 100]



    return render_template('index.html', trending_products=trending_products.head(8),
                           truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))


@app.route("/index")
def indexredirect():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

@app.route('/highly_rated')
def highly_rated():

    # Get top-rated products
    rating_based_rec = rating_based_recommendations(train_data)

    if rating_based_rec.empty:
        message = "No recommendations available for this product."
        return render_template('main.html', message=message)
    else:
        # Create a list of random image URLs for each recommended product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        print(rating_based_rec)
        print(random_product_image_urls)

        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

    # Pass the recommendations to the index.html page
    return render_template('highly_rated.html', rating_based_rec=rating_based_rec,truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))  #This line creates a list of random product image URLs.
                                                               # For each product in trending_products, it picks a random image URL from random_image_urls using random.choice().


@app.route("/signup", methods=['POST', 'GET'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        # Capture the preferred categories from the form
        preferred_categories = request.form.getlist('preferred_categories')  # Get multiple selected categories

        print(f"Received signup data: {username}, {email}, {password}, {preferred_categories}")

        # Convert the list to a comma-separated string for storage
        preferred_categories_str = ', '.join(preferred_categories)
        new_signup = Signup(username=username, email=email, password=password,
                            preferred_categories=preferred_categories_str)
        db.session.add(new_signup)
        db.session.commit()

        return redirect(url_for('profile', user_id=new_signup.id))
        #create a list of random image URLs for each product

        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message=f'{username} signed up successfully '
                               )


#routes for signin page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        new_signup = Signin(username=username, password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Query the database for the user
        user = Signup.query.filter_by(username=username).first()

        # Check if the user exists and the password matches
        if user and user.password == password:
            # Store user ID and other details in the session
            session['user_id'] = user.id
            session['username'] = user.username
            session['email'] = user.email

        # create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message=f'{username} signed in successfully'
                               )


@app.route("/recommendations", methods=['POST', 'GET'])     #content based recommendations
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        nbr = int(request.form.get('nbr'))
        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            message = "No recommendations available for this product."
            return render_template('main.html', message=message, search_query=prod)
        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
            print(content_based_rec)
            print(random_product_image_urls)

            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price),search_query=prod)



@app.route('/main')
def main():
    message = "Search for a product to get recommendations!"
    return render_template('main.html', message=message, content_based_rec=pd.DataFrame())


@app.route('/profile', methods=['GET'])     #collabarative + preffered category
def profile():
        user_id = request.args.get('user_id')
        new_signup = Signup.query.get(user_id)

        if new_signup and new_signup.preferred_categories:
            username = new_signup.username
            email = new_signup.email
            preferred_categories = new_signup.preferred_categories
            if isinstance(preferred_categories, str):
                preferred_categories = [cat.strip() for cat in preferred_categories.split(',')]
            # # If stored as a comma-separated string, convert it back to a list:
            # if isinstance(preferred_categories, str):
            #     preferred_categories = preferred_categories.split(',')

            print("User preferred categories:", preferred_categories)

            # Retrieve products based on the preferred categories
            products = get_products_by_categories(train_data, preferred_categories)

            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]



            if products.empty:
                    message = "No products found for your preferred categories."
                    return render_template('profile.html', username=username, email=email, message=message)


            else:
                    return render_template('profile.html', username=username, email=email, products=products,
                               random_product_image_urls=random_product_image_urls,truncate=truncate,
                               random_price=random.choice(price))

        else:
        #Use the recommendation function to get recommendations for the user

            user_id=session['user_id']
            username=session['username']
            email=session['email']

        # Ensure user_id is properly converted to an integer
        if user_id:
            try:
                user_id = int(user_id)
            except ValueError:
                print("Error: user_id should be an integer.")
                return "Invalid user ID", 400  # Return an error response if invalid

            recommended_products = recommend_items(train_data,user_id, final_ratings_sparse, preds_matrix, num_recommendations=8)

            if recommended_products is None:
                message="User ID not provided"
                return render_template('profile.html', message=message, username=username)

            else:
                # Create a list of random image URLs for each recommended product
                random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
                print(recommended_products)
                print(random_product_image_urls)

                price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]

            return render_template('profile.html',username=username, email=email, recommended_products=recommended_products, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price))



if __name__ == '__main__':
    app.run(debug=True)

