# Module Installation (for Jupyter)

!pip install --upgrade opencv-python 
!pip install torchvision
!pip install tensorflow
!pip install imgaug

# Data Import Libraries/General Lib
import zipfile
import requests
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import copy
import shutil
import io
import cv2
from PIL import Image

# Data Preprocess Libraries
from sklearn.model_selection import cross_val_score, train_test_split
# Data Modelling Libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
# Model Evaluation Libraries 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,precision_score,recall_score,classification_report
# Data Augmentation Libraries
from imgaug import augmenters as iaa
import imgaug as ia

# Target directory to extract the dataset
extract_to = './data'

# Check if the target directory exists, create it if not. This is gonna prevents data loss by not overwriting an existing directory.
if not os.path.exists(extract_to):
    os.makedirs(extract_to)  # Create the directory if it doesn't exist
    print(f"Directory successfully created: {extract_to}")
else:
    print(f"Directory already exists: {extract_to}")  

# Download the dataset zip file from a URL
url = 'https://onedrive.live.com/download?cid=F1ACF0CE4EC03070&resid=F1ACF0CE4EC03070%2166196&authkey=AI5N2M4bGb86aiY'
response = requests.get(url)  

# Extract the zip file contents directly into the target directory without saving the zip file
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(extract_to)
    
# Path to the extracted dataset folder, used for loading and organizing the data
raw_folder = './data/cancer-detection-dataset'

# Clean up: Remove unnecessary folders that are not part of the dataset splits or .DS_Store
exclude_folders = ['.DS_Store', 'cancer-detection-dataset']  # Defined folders to exclude from removal
data_folders = [f for f in os.listdir(extract_to) if os.path.isdir(os.path.join(extract_to, f))]  # List directories

# Loop through directories and remove the ones not needed, in order to keep the workspace clean
for folder in data_folders:
    if folder not in exclude_folders:
        folder_path = os.path.join(extract_to, folder)
        shutil.rmtree(folder_path)  # Remove the folder
        print(f"Unnecessary folder '{folder}' removed successfully.")
print('Your paths succesfully arranged')

# Load the data labels from a CSV file, essential for splitting the dataset according to labels
labels_path=os.path.join(raw_folder, "labels.csv")
labels_df = pd.read_csv(labels_path)

# Function to gather data files that do not end with .csv from raw_folder, For generating image paths
def gather_non_csv_files(raw_folder):
    non_csv_files = []
    for root, dirs, files in os.walk(raw_folder):
        for file in files:
            if not file.endswith('.csv'):
                non_csv_files.append(os.path.join(root, file))
    return non_csv_files

image_paths = gather_non_csv_files(raw_folder)

# DOUBLE-CHECK for IMPORTATION

# List directories and subdirectories in the current working directory
def list_dirs(root_dir):
    """Recursively list all directories and subdirectories under the given root directory."""
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            print(os.path.join(root, dir_name))
# List all 
list_dirs('.')

# Display the first few rows of the LAbel DataFrame
labels_df.head()

# Check if all image paths match with the "id" column in labels_df after removing the .tif extension
image_filenames = [path.split('/')[-1].replace('.tif', '') for path in image_paths]
matching_ids = labels_df['id'].isin(image_filenames).all()
matching_ids # True says noone is left behind, ensuring that we can describe the data using labels.csv dataset

# Length of Image Path
print(f"The number of images in image_paths is {len(image_paths)}.")

## DATA EXPLORATION 

# Get random image ids from the labels DataFrame
sample_ids = labels_df['id'].sample(5)

# Plot images with their labels
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, image_id in zip(axes, sample_ids):
    image_path = os.path.join(raw_folder, f"{image_id}.tif")
    image = Image.open(image_path)
    ax.imshow(image)
    ax.set_title(f"Label: {labels_df[labels_df['id'] == image_id]['label'].values[0]}")
    ax.axis('off')
plt.show()


# Plot the distribution of labels with more descriptive names
# Set the aesthetic style of the plots
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 6))
ax = sns.countplot(x=labels_df['label'].map({0: 'No metastasis', 1: 'Metastasis'}))
plt.title('Distribution of Labels(n=9790)')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks(rotation=0)

# Calculate proportions and add them as annotations inside the bars
total = len(labels_df['label'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    number = f'n={int(p.get_height())}'  # Convert to int to avoid decimals
    x = p.get_x() + p.get_width() / 2
    y = p.get_height() - (0.05 * p.get_height())  # Adjust y position to be inside the bar for percentage
    ax.annotate(percentage, (x, y), size = 20, ha='center', va='top', color='black')
    y_number = p.get_height() - (0.09 * total)  # Adjust y position above the bar for number
    ax.annotate(number, (x, y_number), size = 15, ha='center', va='bottom', color='black')
plt.show()



# DATA SPLITTING

# Split the dataset into training and test sets. Stratification ensures balanced class distribution in each set.
train, test = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['label'])

# Calculate the proportion of each set in the whole dataset
train_proportion = len(train) / len(labels_df)
test_proportion = len(test) / len(labels_df)

# Display the sizes and proportions of each dataset
print(f"Training set size: {len(train)}, Proportion: {train_proportion:.2f}")
print(f"Test set size: {len(test)}, Proportion: {test_proportion:.2f}")

# COPYING Images to relevant directories

# Define the directory structure for the split dataset, including separate folders for training and test sets
data_dir = os.path.join(extract_to, "splitted_data")
train_path = os.path.join(data_dir, 'train')
test_path = os.path.join(data_dir, 'test')

# Create directories for the dataset splits, using exist_ok=True to avoid errors if the directory already exists
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)


# Function to organize files into their corresponding class directories within the dataset splits
def copy_files(df, dst_path):
    """
    Copies files from the source DataFrame (df) to the destination path (dst_path),
    organizing them into subdirectories based on the class label.
    Ensures that the dataset is ready for training with properly organized class directories.
    """
    for _, row in df.iterrows():
        file_name = str(row['id']) + ".tif"  # Knowing that file names are based on 'id' field and have a '.tif' extension
        src_file_path = os.path.join(raw_folder, file_name)
        if not os.path.exists(src_file_path):
            print(f"File not found: {src_file_path}")
            continue  # Skip the file if it's not found, preventing errors
        dst_file_path = os.path.join(dst_path, file_name)  # Define the destination path based on class
        shutil.copy2(src_file_path, dst_file_path)  # Copy the file to its class directory
    print(f'{dst_path} successfully managed')

# Organize files by copying them to their class subdirectories within the split sets
copy_files(train, train_path)
copy_files(test, test_path)

# Define paths
raw_data_dir = raw_folder
train_dir = './data/splitted_data/train'
test_dir = './data/splitted_data/test'

# IMAGE ARRAYING
# Function to load images and convert them into a flat array
def load_images_to_array(directory, dataframe):
    images = []
    labels = []
    for _, row in dataframe.iterrows():
        image_path = os.path.join(directory, row['id'] + '.tif')  
        image = Image.open(image_path)
        image_array = np.array(image).flatten()  # Convert the image to an array and flatten it
        images.append(image_array)
        labels.append(row['label'])
        print(f'{dataframe} successfully arrayed')
    return np.array(images), np.array(labels)

# Load datasets
X_train, y_train = load_images_to_array(train_dir, train)
X_test, y_test = load_images_to_array(test_dir, test)

# MODEL TRAINING+ EVALUATING

def train_evaluate_model(model):
    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred_model = model.predict(X_test)

    # Calculate the accuracy on the test set
    test_accuracy_model = accuracy_score(y_test, y_pred_model)
    
    # Calculate the F1 score on the test set
    f1_score_model = f1_score(y_test, y_pred_model,average='weighted')
    # Calculate AUC scores
    y_pred_scores = model.predict_proba(X_test)[:, 1] 
    auc_score_model = roc_auc_score(y_test, y_pred_scores)
    
    print(f'Model Evaluation without augmentation:')
    print(f'Accuracy: {round(test_accuracy_model, 3)}')
    print(f'F1 Score: {round(f1_score_model, 3)}')
    print(f'AUC: {round(auc_score_model, 3)}')

# 1) GAUSSIAN NAIVE BAYES
nb_classifier = GaussianNB()
train_evaluate_model(nb_classifier)

# 2) GAUSSIAN NAIVE BAYES with K-fold Experiment
# Perform k-fold cross-validation
k = 10  # Number of folds
cv_scores = cross_val_score(nb_classifier, X_train, y_train, cv=k, scoring='accuracy')

# Calculate the average accuracy across all folds
average_cv_accuracy = cv_scores.mean()
average_cv_accuracy


# 3)  RANDOM-FOREST 
rf_classifier=RandomForestClassifier(n_estimators=100, min_samples_split=10, min_samples_leaf=4,random_state=42)
train_evaluate_model(rf_classifier)

# DATA AUGMENTATION EFFECT

def load_and_augment_images(directory, dataframe, augmenter=None):
    images = []
    labels = []
    for _, row in dataframe.iterrows():
        image_path = os.path.join(directory, row['id'] + '.tif')
        image = Image.open(image_path)
        if augmenter is not None:
            image = augmenter.augment_image(np.array(image))
        else:
            image = np.array(image)
        image_array = image.flatten()
        images.append(image_array)
        labels.append(row['label'])
    return np.array(images), np.array(labels)

# Define augmentation techniques
augmentation_techniques = [
    ('No Augmentation', None),
    ('Horizontal Flip', iaa.Fliplr(0.5)),
    ('Vertical Flip', iaa.Flipud(0.5)),
    ('Rotation', iaa.Affine(rotate=(-10, 10))),
    ('Gaussian Blur', iaa.GaussianBlur(sigma=(0, 3.0))),
    ('Brightness', iaa.Multiply((0.8, 1.2))),  # Adjust brightness
    ('Contrast Normalization', iaa.LinearContrast((0.75, 1.5))),  # Adjust contrast
    ('Additive Gaussian Noise', iaa.AdditiveGaussianNoise(scale=(10, 60))),  # Add Gaussian noise
    ('Elastic Transformation', iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25)),  # Apply elastic transformation
    ('Crop and Pad', iaa.CropAndPad(percent=(-0.25, 0.25))),  # Crop/pad images
]

# Iterate over augmentation techniques and train the model object
model= GaussianNB() # We should define our model objects beforehand , replace GaussianNB() with other predefined model objects

# Iterate over augmentation techniques and train the model object
def train_and_evaluate_model_with_augmentation(model):
    """
    Trains and evaluates a given machine learning model using specified data augmentation techniques.
    This function iterates over a list of augmentation techniques, applies them to the training dataset,
    trains the provided model on the augmented training data, and evaluates its performance on the non-augmented test data.
    The accuracy, F1 score, AUC, precision, and recall of the model for each augmentation technique are compiled into a table,
    with all metric values formatted to have 2 decimal points.
    """
    results = []
    metrics = ['accuracy', 'f1', 'auc', 'precision', 'recall']
    for name, augmenter in augmentation_techniques:
        # Load and augment training data
        X_train_aug, y_train_aug = load_and_augment_images(train_dir, train, augmenter)
        # Load but do not augment test data
        X_test, y_test = load_and_augment_images(test_dir, test, None)

        # Train the provided model
        model.fit(X_train_aug, y_train_aug)

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metric_values = {
            'accuracy': round(accuracy_score(y_test, y_pred), 2),
            'f1': round(f1_score(y_test, y_pred), 2),
            'auc': round(roc_auc_score(y_test, y_proba), 2) if y_proba is not None else None,
            'precision': round(precision_score(y_test, y_pred), 2),
            'recall': round(recall_score(y_test, y_pred), 2)
        }

        results.append([name] + [metric_values[metric] for metric in metrics])

    # Create a DataFrame to display results in a table
    results_df = pd.DataFrame(results, columns=['Augmentation Technique'] + metrics)
    display(results_df)

## AUgmentation Heatmaps 
### GNB
data = {
    'Augmentation Technique': [
        'No Augmentation', 'Horizontal Flip', 'Vertical Flip', 'Rotation', 
        'Gaussian Blur', 'Brightness', 'Contrast Normalization', 
        'Additive Gaussian Noise', 'Elastic Transformation', 'Crop and Pad'
    ],
    'Accuracy': [0.75, 0.75, 0.75, 0.73, 0.65, 0.74, 0.75, 0.72, 0.75, 0.68],
    'F1-Score': [0.79, 0.79, 0.79, 0.75, 0.61, 0.79, 0.81, 0.79, 0.78, 0.70],
    'Precision': [0.80, 0.80, 0.80, 0.85, 0.92, 0.78, 0.75, 0.72, 0.81, 0.80],
    'Recall': [0.78, 0.78, 0.78, 0.67, 0.46, 0.80, 0.87, 0.87, 0.77, 0.63]
}

df = pd.DataFrame(data)

# Set the index to be the augmentation techniques
df.set_index('Augmentation Technique', inplace=True)

# Create the heatmap with a fixed color scale from 0.5 to 1
plt.figure(figsize=(12, 8))
sns.heatmap(df, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, annot_kws={"size": 14}, cbar_kws={'label': 'Scale'}, vmin=0.5, vmax=1)
plt.title('GNB-Performance Metrics Across Augmentation Techniques')
plt.xlabel('Metrics', fontsize=14) 
plt.ylabel('Augmentation Technique', fontsize=14)  
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12) 
plt.show()

### RF 
data = {
    'Augmentation Technique': [
        'No Augmentation', 'Horizontal Flip', 'Vertical Flip', 'Rotation', 
        'Gaussian Blur', 'Brightness', 'Contrast Normalization', 
        'Additive Gaussian Noise', 'Elastic Transformation', 'Crop and Pad'
    ],
    'Accuracy': [0.76, 0.76, 0.76, 0.77, 0.74, 0.76, 0.75, 0.73, 0.76, 0.76],
    'F1-Score': [0.81, 0.81, 0.81, 0.80, 0.77, 0.82, 0.81, 0.80, 0.81, 0.80],
    'Precision': [0.78, 0.77, 0.77, 0.82, 0.82, 0.76, 0.75, 0.72, 0.78, 0.81],
    'Recall': [0.85, 0.85, 0.86, 0.78, 0.73, 0.88, 0.88, 0.90, 0.85, 0.79]
}

df = pd.DataFrame(data)

# Set the index to be the augmentation techniques
df.set_index('Augmentation Technique', inplace=True)

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, annot_kws={"size": 14}, cbar_kws={'label': 'Scale'}, vmin=0.5, vmax=1)
plt.title('RF-Performance Metrics Across Augmentation Techniques')
plt.xlabel('Metrics', fontsize=14)  
plt.ylabel('Augmentation Technique', fontsize=14) 
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12) 
plt.show()




def evaluate_model_performance(model):
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate AUC scores
    y_pred_scores = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_scores)

    # Generate a classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # Extracting performance metrics
    accuracy = report['accuracy']
    f1_score = report['macro avg']['f1-score']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']

    # Displaying the results
    print(f'{model.__class__.__name__} Evaluation:')
    print(f'Accuracy: {accuracy:.3f}')
    print(f'F1 Score: {f1_score:.3f}')
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'AUC: {auc_score:.3f}')


# COMPARING MODELS

# We should define our model objects beforehand , replace model objects with other predefined model objects
model1= nb_classifier
model2= rf_classifier

def plot_roc_curve(model, name='Model', color='blue', linestyle='-'):
    probs = model.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color=color, linestyle=linestyle, lw=2, label=f'{name} (Area Under Curve = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Ideal point (0,1) with a star marker
    plt.plot(0, 1, marker='*', markersize=10, color="red", label='Ideal Point')
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'Receiver Operating Characteristic - {name}',fontsize=18)
    plt.legend(loc="lower right", fontsize=16)
    plt.show()