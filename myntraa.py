#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import zipfile

# Define the path to the ZIP file
zip_path = 'archive (1).zip'
csv_filename = 'Fashion Dataset.csv'

# Extract the CSV file from the ZIP archive and load it into a DataFrame
with zipfile.ZipFile(zip_path, 'r') as z:
    with z.open(csv_filename) as f:
        shopping_data = pd.read_csv(f)

# Print the first few rows of the DataFrame
print(shopping_data.head())


# In[2]:


import pandas as pd
import zipfile
import os

# Define the path to the ZIP file
zip_path = 'archive (1).zip'

# Create a directory to extract the files
extracted_path = 'extracted_files'
os.makedirs(extracted_path, exist_ok=True)

# Extract all contents of the ZIP file
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extracted_path)

# Define the paths to the extracted CSV file and image folder
csv_path = os.path.join(extracted_path, 'Fashion Dataset.csv')
image_folder_path = os.path.join(extracted_path, 'Images')  # Replace with actual folder name

# Load the CSV file into a DataFrame
fashion_data = pd.read_csv(csv_path)
print(fashion_data.head())

# List all images in the image folder
image_files = os.listdir(image_folder_path)
print("Image files:", image_files)

# Now you can process the image files as needed
# For example, you can load images using a library like PIL or OpenCV


# In[3]:


import zipfile
import os

# Define the path to the ZIP file
zip_path = 'C://Users/riyau/Downloads/archive (1).zip'

# Create a directory to extract the files
extracted_path = 'extracted_files'
os.makedirs(extracted_path, exist_ok=True)

# Extract all contents of the ZIP file
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall(extracted_path)

# List all files and directories in the extracted path
extracted_contents = os.listdir(extracted_path)
print("Extracted contents:", extracted_contents)


# In[4]:


from PIL import Image

# Define the path to the images folder within the extracted directory
image_folder_path = os.path.join(extracted_path, 'Images', 'Images')

# Check if the folder exists
if os.path.exists(image_folder_path) and os.path.isdir(image_folder_path):
    # List all images in the image folder
    image_files = os.listdir(image_folder_path)
    print("Image files:", image_files)

    # Check if there are any image files
    if image_files:
        # Example: Open and display the first image using PIL
        first_image_path = os.path.join(image_folder_path, image_files[3])
        image = Image.open(first_image_path)
        image.show()

        # Example: Convert the image to grayscale
        gray_image = image.convert('L')
        gray_image.show()
    else:
        print("No image files found in the specified folder.")
else:
    print("The specified images folder does not exist.")


# In[5]:


import pandas as pd
import zipfile

# Specify the path to your zip file
zip_file_path = 'C://Users/riyau/Downloads/archive (1).zip'

# Extract the CSV file from the zip archive
with zipfile.ZipFile(zip_file_path, 'r') as z:
    # Assuming 'Fashion Dataset.csv' is the name of the CSV file inside the zip
    csv_file = z.open('Fashion Dataset.csv')
    df = pd.read_csv(csv_file)

# Display the first few rows and columns to understand the structure
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Preprocess text data if needed (description column)
# Example: Convert description to lowercase
df['description'] = df['description'].str.lower()

# Print basic statistics or information about the dataset
print(df.info())


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF Vectorization of product descriptions
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'].fillna(''))

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get recommendations based on product name
indices = pd.Series(df.index, index=df['name']).drop_duplicates()

def get_recommendations(name, cosine_sim=cosine_sim):
    idx = indices[name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Top 5 similar products
    product_indices = [i[0] for i in sim_scores]
    return df['name'].iloc[product_indices]



# # Top 10 recommendations for You

# In[9]:


from IPython.display import display, Image

print("Top 10 Recommended Products:")
for index, row in df.head(10).iterrows():
    print(f"{row['name']} (Avg. Rating: {row['avg_rating']:.2f})")
    display(Image(url=row['img'], width=200, height=200))
    print()


# In[ ]:




