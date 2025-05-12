# Kaspi Market Sales Prediction

This web application predicts sales for products on Kaspi Market using both supervised and unsupervised machine learning algorithms.

## Features

- Supervised Learning: Random Forest Regressor for sales prediction
- Unsupervised Learning: K-means clustering for product segmentation
- Modern web interface with Bootstrap
- Real-time predictions
- Model training through CSV file upload

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. To train the models:
   - Prepare a CSV file with the following columns:
     - price: Product price
     - views: Number of views
     - likes: Number of likes
     - comments: Number of comments
     - sales: Actual sales (for supervised learning)
   - Upload the CSV file using the "Train Models" section

4. To make predictions:
   - Enter the product details (price, views, likes, comments)
   - Click "Predict Sales"
   - View the prediction results:
     - Supervised prediction: Expected number of sales
     - Cluster assignment: Product segment (0, 1, or 2)

## Data Format

The training CSV file should have the following format:
```csv
price,views,likes,comments,sales
1000,500,50,10,5
2000,1000,100,20,10
...
```

## Models

1. Supervised Learning (Random Forest):
   - Predicts the number of sales based on product features
   - Uses price, views, likes, and comments as input features

2. Unsupervised Learning (K-means):
   - Clusters products into 3 segments based on their features
   - Helps identify product categories and patterns

## Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- Other dependencies listed in requirements.txt 