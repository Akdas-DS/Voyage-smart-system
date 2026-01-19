import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Voyage | Travel Intelligence System",
    page_icon="✈️",
    layout="wide"
)

# -------------------------------------------------
# Custom styling (human, minimal, professional)
# -------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    h1, h2, h3 {
        color: #1f2a44;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    .recommendation-card {
        padding: 1rem;
        border-radius: 8px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Load data and model
# -------------------------------------------------
@st.cache_data
def load_data():
    flights = pd.read_csv("data/flights.csv")
    users = pd.read_csv("data/users.csv")
    hotels = pd.read_csv("data/hotels.csv")
    
    # Convert date column if exists
    if 'date' in hotels.columns:
        hotels['date'] = pd.to_datetime(hotels['date'])
    
    return flights, users, hotels

@st.cache_resource
def load_model():
    model = joblib.load("models/flight_price_model.pkl")
    columns = joblib.load("models/flight_columns.pkl")
    gender_model = joblib.load("models/gender_model.pkl")
    return model, columns, gender_model


flights_df, users_df, hotels_df = load_data()
model, model_columns, gender_model = load_model()


# -------------------------------------------------
# Precompute features for hybrid recommendation
# -------------------------------------------------
@st.cache_data
def precompute_features():
    """Precompute all features needed for hybrid recommendation"""
    
    # 1. Hotel similarity features (current model)
    hotel_features = (
        hotels_df
        .groupby("name")
        .agg(
            place=("place", "first"),
            avg_price=("price", "mean"),
            popularity=("name", "count")
        )
        .reset_index()
    )
    
    # Preprocessor for similarity
    preprocessor = ColumnTransformer(
        transformers=[
            ("place", OneHotEncoder(handle_unknown="ignore"), ["place"]),
            ("num", StandardScaler(), ["avg_price", "popularity"])
        ]
    )
    
    hotel_matrix = preprocessor.fit_transform(hotel_features)
    similarity = cosine_similarity(hotel_matrix)
    
    hotel_index = {name: idx for idx, name in enumerate(hotel_features["name"])}
    
    # 2. User profile features
    # Ensure date column exists and is datetime
    if 'date' in hotels_df.columns and 'userCode' in hotels_df.columns:
        user_profile = (
            hotels_df
            .groupby('userCode')
            .agg(
                total_trips=('name', 'count'),
                total_spend=('price', 'sum'),
                avg_days=('days', 'mean'),
                last_trip=('date', 'max')
            )
            .reset_index()
        )
        
        # Calculate recency days safely
        if not hotels_df['date'].empty:
            user_profile['recency_days'] = (
                hotels_df['date'].max() - user_profile['last_trip']
            ).dt.days
    else:
        user_profile = pd.DataFrame(columns=['userCode'])
    
    # Popular hotels
    if 'name' in hotels_df.columns:
        popular_hotels = (
            hotels_df
            .groupby('name')
            .agg(
                total_bookings=('name', 'count'),
                avg_price=('price', 'mean'),
                place=('place', 'first')
            )
            .sort_values('total_bookings', ascending=False)
        )
    else:
        popular_hotels = pd.DataFrame()
    
    # 3. Budget-based features
    if 'name' in hotels_df.columns and 'days' in hotels_df.columns:
        hotel_price_days_df = (
            hotels_df
            .groupby('name')
            .agg(
                avg_price=('price', 'mean'),
                avg_days=('days', 'mean'),
                min_price=('price', 'min'),
                max_price=('price', 'max'),
                popularity=('name', 'count'),
                place=('place', 'first')
            )
            .reset_index()
        )
    else:
        hotel_price_days_df = pd.DataFrame()
    
    # Scaler for budget-based recommendations
    if not hotel_price_days_df.empty and 'avg_price' in hotel_price_days_df.columns and 'avg_days' in hotel_price_days_df.columns:
        scaler = StandardScaler()
        hotel_scaled_features = scaler.fit_transform(
            hotel_price_days_df[['avg_price', 'avg_days']]
        )
    else:
        scaler = None
        hotel_scaled_features = None
    
    return {
        'hotel_features': hotel_features,
        'similarity_matrix': similarity,
        'hotel_index': hotel_index,
        'user_profile': user_profile,
        'popular_hotels': popular_hotels,
        'hotel_price_days_df': hotel_price_days_df,
        'scaler': scaler,
        'hotel_scaled_features': hotel_scaled_features
    }

# -------------------------------------------------
# Recommendation Functions
# -------------------------------------------------
def recommend_popular_hotels(user_id=None, top_n=5):
    """Recommend hotels based on popularity"""
    features = precompute_features()
    popular_hotels = features['popular_hotels']
    
    if popular_hotels.empty:
        return pd.DataFrame()
    
    # If user_id provided, exclude hotels they've visited
    if user_id is not None and 'userCode' in hotels_df.columns and 'name' in hotels_df.columns:
        visited_hotels = set(
            hotels_df[hotels_df['userCode'] == user_id]['name']
        )
        recommendations = popular_hotels[
            ~popular_hotels.index.isin(visited_hotels)
        ].head(top_n)
    else:
        recommendations = popular_hotels.head(top_n)
    
    return recommendations.reset_index()

def recommend_similar_hotels(hotel_name, top_n=5):
    """Recommend hotels similar to given hotel"""
    features = precompute_features()
    
    if hotel_name not in features['hotel_index']:
        return pd.DataFrame()
    
    idx = features['hotel_index'][hotel_name]
    scores = features['similarity_matrix'][idx]
    top_idx = scores.argsort()[::-1][1:top_n+1]
    
    recommendations = features['hotel_features'].iloc[top_idx][
        ["name", "place", "avg_price"]
    ]
    recommendations = recommendations.rename(columns={'name': 'hotel_name'})
    return recommendations

def recommend_hotels_by_budget_and_days(user_budget, user_days, top_n=5):
    """Recommend hotels based on budget and duration"""
    features = precompute_features()
    hotel_price_days_df = features['hotel_price_days_df']
    scaler = features['scaler']
    
    if hotel_price_days_df.empty or scaler is None:
        return pd.DataFrame()
    
    # Create user preference vector
    user_vector = scaler.transform([[user_budget, user_days]])
    
    # Compute distance
    distances = euclidean_distances(
        user_vector,
        features['hotel_scaled_features']
    )[0]
    
    hotel_price_days_df['distance'] = distances
    
    # Rank by closest match + popularity boost
    recommendations = (
        hotel_price_days_df
        .sort_values(['distance', 'popularity'], ascending=[True, False])
        .head(top_n)
    )
    
    return recommendations[
        ['name', 'avg_price', 'avg_days', 'popularity', 'place', 'distance']
    ].rename(columns={'name': 'hotel_name'})

def hybrid_recommendation_system(
    user_id=None,
    hotel_name=None,
    user_budget=None,
    user_days=None,
    top_n=5,
    weights=None,
    return_details=False
):
    """
    Hybrid recommendation system combining:
    1. Popular hotels (for new/cold-start users)
    2. Similar hotels (for users with specific preferences)
    3. Budget-based hotels (for practical recommendations)
    """
    
    # Default weights if not provided
    if weights is None:
        weights = {
            'popularity': 0.3,
            'similarity': 0.4,
            'budget': 0.3
        }
    
    # Normalize weights to sum to 1
    weight_sum = sum(weights.values())
    weights = {k: v/weight_sum for k, v in weights.items()}
    
    all_recommendations = {}
    score_details = {}
    
    # 1. Get popular hotel recommendations (if user_id is provided)
    if user_id is not None:
        try:
            pop_recs = recommend_popular_hotels(user_id, top_n=top_n*3)
            if not pop_recs.empty:
                # Assign scores based on popularity rank
                pop_recs['popularity_score'] = np.linspace(
                    weights['popularity'], 
                    weights['popularity'] * 0.1, 
                    len(pop_recs)
                )
                
                for _, row in pop_recs.iterrows():
                    hotel = row['name'] if 'name' in row else row['hotel_name']
                    score = row['popularity_score']
                    
                    if hotel not in all_recommendations:
                        all_recommendations[hotel] = 0
                        score_details[hotel] = {'popularity': 0, 'similarity': 0, 'budget': 0}
                    
                    all_recommendations[hotel] += score
                    score_details[hotel]['popularity'] = score
        except Exception as e:
            st.warning(f"Popularity recommendation skipped: {e}")
    
    # 2. Get similar hotel recommendations (if hotel_name is provided)
    if hotel_name is not None:
        try:
            sim_recs = recommend_similar_hotels(hotel_name, top_n=top_n*3)
            if not sim_recs.empty:
                # Similarity scores are based on rank
                sim_recs['similarity_score'] = np.linspace(
                    weights['similarity'], 
                    weights['similarity'] * 0.1, 
                    len(sim_recs)
                )
                
                for idx, row in sim_recs.iterrows():
                    hotel = row['hotel_name']
                    score = row['similarity_score']
                    
                    if hotel not in all_recommendations:
                        all_recommendations[hotel] = 0
                        score_details[hotel] = {'popularity': 0, 'similarity': 0, 'budget': 0}
                    
                    all_recommendations[hotel] += score
                    score_details[hotel]['similarity'] = score
        except Exception as e:
            st.warning(f"Similarity recommendation skipped: {e}")
    
    # 3. Get budget-based recommendations
    if user_budget is not None and user_days is not None:
        try:
            budget_recs = recommend_hotels_by_budget_and_days(
                user_budget, 
                user_days, 
                top_n=top_n*3
            )
            if not budget_recs.empty:
                # Budget score based on inverse distance (closer = higher score)
                max_dist = budget_recs['distance'].max()
                if max_dist > 0:
                    budget_recs['budget_score'] = weights['budget'] * (
                        1 - budget_recs['distance'] / max_dist
                    )
                else:
                    budget_recs['budget_score'] = weights['budget']
                
                for _, row in budget_recs.iterrows():
                    hotel = row['hotel_name']
                    score = row['budget_score']
                    
                    if hotel not in all_recommendations:
                        all_recommendations[hotel] = 0
                        score_details[hotel] = {'popularity': 0, 'similarity': 0, 'budget': 0}
                    
                    all_recommendations[hotel] += score
                    score_details[hotel]['budget'] = score
        except Exception as e:
            st.warning(f"Budget recommendation skipped: {e}")
    
    # If no specific inputs, fall back to pure popularity
    if not all_recommendations and user_id is None and hotel_name is None:
        try:
            features = precompute_features()
            popular_hotels = features['popular_hotels']
            pop_recs = popular_hotels.head(top_n).reset_index()
            pop_recs = pop_recs.rename(columns={'name': 'hotel_name'})
            if return_details:
                return pop_recs, None
            return pop_recs
        except:
            # Return empty if everything fails
            return pd.DataFrame(columns=['hotel_name', 'place', 'avg_price', 'total_score'])
    
    # Convert to DataFrame and sort by total score
    recommendations = []
    for hotel, total_score in all_recommendations.items():
        # Get hotel details from original dataset
        hotel_data = hotels_df[hotels_df['name'] == hotel]
        if not hotel_data.empty:
            hotel_details = hotel_data.iloc[0]
            
            rec = {
                'hotel_name': hotel,
                'total_score': total_score,
                'popularity_score': score_details[hotel]['popularity'],
                'similarity_score': score_details[hotel]['similarity'],
                'budget_score': score_details[hotel]['budget'],
                'place': hotel_details.get('place', 'Unknown'),
                'avg_price': hotel_details.get('price', 0),
                'popularity': score_details[hotel]['popularity'] + score_details[hotel]['similarity'] + score_details[hotel]['budget']
            }
            recommendations.append(rec)
    
    recommendations_df = pd.DataFrame(recommendations)
    
    if not recommendations_df.empty:
        # Sort by total score
        recommendations_df = recommendations_df.sort_values(
            'total_score', 
            ascending=False
        ).head(top_n).reset_index(drop=True)
    
        if return_details:
            return recommendations_df, score_details
        
        return recommendations_df[['hotel_name', 'place', 'avg_price', 'total_score']]
    
    return pd.DataFrame(columns=['hotel_name', 'place', 'avg_price', 'total_score'])

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("✈️ Voyage")
st.sidebar.caption("Travel Price Prediction & Recommendation")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Flight Price Prediction",
        "Hotel Recommendation",
        "Gender Classification"
    ]
)


# -------------------------------------------------
# Overview Page
# -------------------------------------------------
if page == "Overview":
    st.title("Travel Intelligence Dashboard")

    st.write(
        "This application helps users **predict flight prices** and "
        "**discover similar hotels** using machine learning models."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Flights Records", flights_df.shape[0])
    col2.metric("Users Records", users_df.shape[0])
    col3.metric("Hotels Records", hotels_df.shape[0])

    st.subheader("Sample Data")
    st.dataframe(flights_df.head())

# -------------------------------------------------
# Flight Price Prediction Page
# -------------------------------------------------
elif page == "Flight Price Prediction":
    st.title("✈️ Flight Price Prediction")

    st.write("Enter flight details to estimate the expected ticket price.")

    col1, col2 = st.columns(2)

    with col1:
        distance = st.number_input("Distance (km)", min_value=100, max_value=20000, step=50)
        time = st.number_input("Flight Time (minutes)", min_value=30, max_value=1500, step=10)
        agency = st.selectbox("Agency", flights_df["agency"].unique())

    with col2:
        flight_type = st.selectbox("Flight Type", flights_df["flightType"].unique())
        source = st.selectbox("From", flights_df["from"].unique())
        destination = st.selectbox("To", flights_df["to"].unique())

    if st.button("Predict Price"):
        input_df = pd.DataFrame([{
            "distance": distance,
            "time": time,
            "agency": agency,
            "flightType": flight_type,
            "from": source,
            "to": destination
        }])

        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]

        st.success(f"Estimated Flight Price: ₹ {int(prediction):,}")

# -------------------------------------------------
# Hotel Recommendation Page
# -------------------------------------------------
elif page == "Hotel Recommendation":
    st.title("🏨 Hybrid Hotel Recommendation System")
    
    st.write("""
    Get personalized hotel recommendations using our hybrid system that combines:
    1. **Popularity**: Most booked hotels
    2. **Similarity**: Hotels similar to ones you like
    3. **Budget**: Hotels matching your budget and stay duration
    """)
    
    # Precompute features
    features = precompute_features()
    
    # Create tabs for different recommendation methods
    tab1, tab2, tab3, tab4 = st.tabs([
        "Hybrid Recommendations", 
        "Popularity Based",
        "Similarity Based", 
        "Budget Based"
    ])
    
    with tab1:
        st.subheader("Hybrid Recommendations")
        st.write("Get personalized recommendations combining multiple factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get user IDs from data
            user_ids = []
            if 'userCode' in hotels_df.columns:
                user_ids = hotels_df['userCode'].unique().tolist()
            
            user_id = st.selectbox(
                "User ID (Optional)",
                options=["None"] + user_ids,
                index=0
            )
            
            # Get available hotels
            available_hotels = features['hotel_features']['name'].unique()
            hotel_name = st.selectbox(
                "Hotel you like (Optional)",
                options=["None"] + list(available_hotels)
            )
        
        with col2:
            user_budget = st.number_input(
                "Your budget per night (₹)", 
                min_value=0, 
                max_value=100000,
                value=5000,
                step=500
            )
            user_days = st.number_input(
                "Number of days", 
                min_value=1, 
                max_value=30,
                value=3,
                step=1
            )
        
        # Weight controls
        st.subheader("Adjust Recommendation Weights")
        col3, col4, col5 = st.columns(3)
        with col3:
            pop_weight = st.slider("Popularity Weight", 0.0, 1.0, 0.3, 0.05)
        with col4:
            sim_weight = st.slider("Similarity Weight", 0.0, 1.0, 0.4, 0.05)
        with col5:
            bud_weight = st.slider("Budget Weight", 0.0, 1.0, 0.3, 0.05)
        
        weights = {
            'popularity': pop_weight,
            'similarity': sim_weight,
            'budget': bud_weight
        }
        
        if st.button("Get Hybrid Recommendations"):
            with st.spinner("Finding the best hotels for you..."):
                recommendations = hybrid_recommendation_system(
                    user_id=user_id if user_id != "None" else None,
                    hotel_name=hotel_name if hotel_name != "None" else None,
                    user_budget=user_budget,
                    user_days=user_days,
                    top_n=10,
                    weights=weights
                )
                
                if not recommendations.empty:
                    st.success(f"Found {len(recommendations)} recommendations for you!")
                    
                    # Display with better formatting
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            col_a, col_b, col_c = st.columns([3, 1, 1])
                            with col_a:
                                st.markdown(f"**{row['hotel_name']}**")
                                st.caption(f"📍 {row['place']}")
                            with col_b:
                                st.metric("Avg Price", f"₹{row['avg_price']:,.0f}")
                            with col_c:
                                st.metric("Score", f"{row['total_score']:.2f}")
                            st.divider()
                else:
                    st.warning("No recommendations found. Try adjusting your criteria.")
    
    with tab2:
        st.subheader("Popularity Based Recommendations")
        st.write("Most booked hotels by all users")
        
        top_n = st.slider("Number of recommendations", 5, 20, 10, key="pop_slider")
        
        if st.button("Show Popular Hotels"):
            recommendations = recommend_popular_hotels(top_n=top_n)
            
            if not recommendations.empty:
                st.dataframe(
                    recommendations[['name', 'place', 'avg_price', 'total_bookings']]
                    .rename(columns={'name': 'Hotel', 'place': 'Location', 'avg_price': 'Avg Price', 'total_bookings': 'Total Bookings'})
                )
            else:
                st.warning("Could not generate popularity recommendations.")
    
    with tab3:
        st.subheader("Similarity Based Recommendations")
        st.write("Find hotels similar to ones you like")
        
        selected_hotel = st.selectbox(
            "Choose a hotel you like",
            features['hotel_features']['name'].unique(),
            key="similarity_hotel"
        )
        
        top_n = st.slider("Number of recommendations", 5, 20, 10, key="sim_slider")
        
        if st.button("Find Similar Hotels"):
            recommendations = recommend_similar_hotels(selected_hotel, top_n=top_n)
            
            if not recommendations.empty:
                st.dataframe(
                    recommendations.rename(columns={'hotel_name': 'Hotel', 'place': 'Location', 'avg_price': 'Avg Price'})
                )
            else:
                st.warning("Could not generate similarity recommendations.")
    
    with tab4:
        st.subheader("Budget Based Recommendations")
        st.write("Find hotels matching your budget and stay duration")
        
        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input(
                "Budget per night (₹)", 
                min_value=0, 
                max_value=100000,
                value=5000,
                step=500,
                key="budget_input"
            )
        with col2:
            days = st.number_input(
                "Number of days", 
                min_value=1, 
                max_value=30,
                value=3,
                step=1,
                key="days_input"
            )
        
        top_n = st.slider("Number of recommendations", 5, 20, 10, key="budget_slider")
        
        if st.button("Find Budget Hotels"):
            recommendations = recommend_hotels_by_budget_and_days(budget, days, top_n=top_n)
            
            if not recommendations.empty:
                st.dataframe(
                    recommendations[['hotel_name', 'place', 'avg_price', 'avg_days', 'distance']]
                    .rename(columns={
                        'hotel_name': 'Hotel',
                        'place': 'Location',
                        'avg_price': 'Avg Price',
                        'avg_days': 'Avg Stay (days)',
                        'distance': 'Match Score'
                    })
                )
            else:
                st.warning("Could not generate budget-based recommendations.")
    
    # Display user profiles if available
    if 'userCode' in hotels_df.columns:
        with st.expander("View User Profiles"):
            features = precompute_features()
            user_profile = features['user_profile']
            
            if not user_profile.empty:
                st.dataframe(user_profile.head(10))
            else:
                st.info("User profile data not available.")

elif page == "Gender Classification":
    st.title("👤 Gender Classification")

    st.write(
        "This module predicts gender using linguistic patterns in names, "
        "organizational context, and age. The model is trained using "
        "character-level text features for higher accuracy."
    )

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Full Name")
        company = st.text_input("Organization / Company")

    with col2:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=100,
            value=25,
            step=1
        )

    if st.button("Predict Gender"):
        if not name.strip() or not company.strip():
            st.warning("Please enter both name and organization.")
        else:
            input_df = pd.DataFrame([{
                "name": name,
                "company": company,
                "age": age
            }])

            prediction = gender_model.predict(input_df)[0]
            st.success(f"Predicted Gender: {prediction.capitalize()}")

# -------------------------------------------------
# Run the app
# -------------------------------------------------
if __name__ == "__main__":
    pass