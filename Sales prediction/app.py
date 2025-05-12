from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

app = Flask(__name__)

# Initialize models
global_models = {}
unsupervised_model = None
scaler = StandardScaler()
model_scores = {}
last_training_data = None
last_cluster_labels = None

# List of models to train
regressors = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(),
    'SVM': SVR(),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

# For demonstration, Logistic Regression and Naive Bayes as classifiers (if sales is binary)
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB()
}

def train_models(data):
    global global_models, unsupervised_model, scaler, model_scores, last_training_data, last_cluster_labels
    global_models = {}
    model_scores = {}
    last_training_data = data.copy()
    # Prepare features for supervised learning
    X = data[['price', 'views', 'likes', 'comments']]
    y = data['sales']
    is_binary = y.nunique() == 2
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Dynamically set n_neighbors for KNN
    n_neighbors = min(5, len(X_train))
    local_regressors = regressors.copy()
    local_regressors['KNN'] = KNeighborsRegressor(n_neighbors=n_neighbors)
    # GridSearchCV parameter grids
    param_grids = {
        'Linear Regression': {},
        'Decision Tree': {'max_depth': [None, 2, 3, 4, 5]},
        'Random Forest': {'n_estimators': [50, 100], 'max_depth': [None, 3, 5]},
        'KNN': {'n_neighbors': list(range(1, min(6, len(X_train)) + 1))},
        'SVM': {'C': [0.1, 1, 10]},
        'Gradient Boosting': {'n_estimators': [50, 100], 'max_depth': [2, 3, 4]}
    }
    for name, model in local_regressors.items():
        grid = GridSearchCV(model, param_grids.get(name, {}), cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        y_pred = np.maximum(y_pred, 0)  # No negative predictions
        score = r2_score(y_test, y_pred)
        # Cross-validated score
        cv_score = cross_val_score(best_model, X_train_scaled, y_train, cv=3, scoring='r2').mean()
        model_scores[name] = {'test_r2': score, 'cv_r2': cv_score}
        global_models[name] = best_model
    # Train classifiers if binary
    if is_binary:
        for name, model in classifiers.items():
            grid = GridSearchCV(model, {}, cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train_scaled, y_train)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test_scaled)
            score = (y_pred == y_test).mean()
            cv_score = cross_val_score(best_model, X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
            model_scores[name] = {'test_acc': score, 'cv_acc': cv_score}
            global_models[name] = best_model
    # Train unsupervised model (K-means)
    unsupervised_model = KMeans(n_clusters=3, random_state=42)
    unsupervised_model.fit(X_train_scaled)
    # Save cluster labels for all data (using all data, not just train)
    all_scaled = scaler.transform(data[['price', 'views', 'likes', 'comments']])
    last_cluster_labels = unsupervised_model.predict(all_scaled)
    # Save models
    for name, model in global_models.items():
        joblib.dump(model, f'{name.replace(" ", "_").lower()}_model.joblib')
    joblib.dump(unsupervised_model, 'unsupervised_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    # Save performance graph
    plt.figure(figsize=(10,5))
    bar_labels = []
    bar_values = []
    for name, scores in model_scores.items():
        if 'test_r2' in scores:
            bar_labels.append(f'{name} (R2)')
            bar_values.append(scores['test_r2'])
        elif 'test_acc' in scores:
            bar_labels.append(f'{name} (Acc)')
            bar_values.append(scores['test_acc'])
    plt.bar(bar_labels, bar_values, color='skyblue')
    plt.ylabel('R2 Score / Accuracy')
    plt.title('Model Performance (Test Set)')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig('static/model_performance.png')
    plt.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global global_models, unsupervised_model, scaler
    if not global_models or unsupervised_model is None:
        return jsonify({'error': 'Models are not trained yet. Please upload training data first.'}), 400
    data = request.get_json()
    try:
        input_data = np.array([[
            float(data['price']),
            float(data['views']),
            float(data['likes']),
            float(data['comments'])
        ]])
        input_scaled = scaler.transform(input_data)
        predictions = {}
        for name, model in global_models.items():
            pred = model.predict(input_scaled)[0]
            predictions[name] = float(pred)
        cluster = unsupervised_model.predict(input_scaled)[0]
        return jsonify({
            'predictions': predictions,
            'cluster': int(cluster)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    try:
        data = pd.read_csv(file)
        train_models(data)
        return jsonify({'message': 'Models trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/performance-graph')
def performance_graph():
    return send_file('static/model_performance.png', mimetype='image/png')

@app.route('/model-metrics')
def model_metrics():
    global model_scores
    return jsonify(model_scores)

@app.route('/clusters')
def clusters():
    global last_training_data, last_cluster_labels
    if last_training_data is None or last_cluster_labels is None:
        return jsonify({'error': 'No clustering info available. Train the model first.'}), 400
    clusters_info = {}
    df = last_training_data.copy()
    df['cluster'] = last_cluster_labels
    for cluster_id in sorted(df['cluster'].unique()):
        # Show up to 5 samples from each cluster
        samples = df[df['cluster'] == cluster_id].head(5).to_dict(orient='records')
        clusters_info[int(cluster_id)] = samples
    return jsonify(clusters_info)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True) 