import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


users_df = pd.read_csv("data/users.csv")

# Keep required columns
users_df = users_df[["name", "company", "age", "gender"]]

# Drop missing values (safety)
users_df = users_df.dropna()

X = users_df[["name", "company", "age"]]
y = users_df["gender"]


text_features = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=3
        )
    )
])

preprocessor = ColumnTransformer(
    transformers=[
        ("name_tfidf", text_features, "name"),
        ("company_tfidf", text_features, "company"),
        ("age_scaler", StandardScaler(), ["age"])
    ]
)


model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            multi_class="auto"
        )
    )
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Gender Classification Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


joblib.dump(model, "models/gender_model.pkl")
