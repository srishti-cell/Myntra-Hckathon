#!/usr/bin/env python
# coding: utf-8

# # Trending Maharastrian Culture

# In[ ]:


import pandas as pd
import zipfile
import random
import IPython.display as display

# Define the path to the ZIP file and CSV filename
zip_path = 'archive (1).zip'
csv_filename = 'Fashion Dataset.csv'

# Extract the CSV file from the ZIP archive and load it into a DataFrame
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        shopping_data = pd.read_csv(f)

# Display the first few rows to understand the structure
shopping_data.head()

# Function to generate trend recommendations (mock example)
def generate_trend_recommendations(num_recommendations):
    # Mock example - randomly select items from the dataset
    recommendations = shopping_data.sample(n=num_recommendations)
    return recommendations

# Function to display trend recommendations with images
def display_trend_recommendations(recommendations):
    for index, row in recommendations.iterrows():
        display.display(display.HTML(f"<h2>{row['name']}</h2><p>{row['description']}</p>"))
        display.display(display.Image(url=row['img']))

# Generate trend recommendations (replace with AI-powered logic as needed)
num_recommendations = 9
trend_recommendations = generate_trend_recommendations(num_recommendations)

# Display trend recommendations with images
display_trend_recommendations(trend_recommendations)


