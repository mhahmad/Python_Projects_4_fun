import os  # Importing the operating system module
import glob  # Importing the module to retrieve files/pathnames matching a specified pattern
import torch  # Importing the PyTorch library for machine learning
from PIL import Image  # Importing the Python Imaging Library for image processing
import torchvision.transforms as transforms  # Importing transformations for images in PyTorch
import clip  # Importing the CLIP (Contrastive Language-Image Pre-training) model
import matplotlib.pyplot as plt  # Importing the plotting library
import pickle  # Importing the module for serializing and deserializing Python objects

# Folder where images are stored
image_folder = "./images"

# Check if CUDA (GPU) is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model (ResNet-50 architecture) and set it to the appropriate device
clip_model, _ = clip.load("RN50", device=device)

# Function to preprocess an image
def preprocess_image(image_path):
    """
    Preprocesses an image:
    1. Opens the image file.
    2. Converts image mode to RGB if it's RGBA.
    3. Resizes and crops the image to 224x224 pixels.
    4. Converts the image to a PyTorch tensor.
    5. Normalizes the image tensor.
    """
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize image
    ])
    
    image = preprocess(image).unsqueeze(0)  # Add a batch dimension
    return image

# Function to create an index of images with their CLIP feature embeddings
def create_image_index(image_folder, clip_model):
    """
    Creates a dictionary of image paths mapped to their CLIP feature embeddings:
    1. Finds image files in the specified folder.
    2. Processes each image to obtain its CLIP feature embeddings.
    3. Stores the mappings in a dictionary.
    4. Saves the dictionary to a file for future use.
    """
    index = {}  # Dictionary for (image_path:image_features)
    images = glob.glob(os.path.join(image_folder, "*.jpg")) + glob.glob(os.path.join(image_folder, "*.jpeg")) + glob.glob(os.path.join(image_folder, "*.png"))

    # Load existing index if available
    with open('data.pkl', 'rb') as file:
        old_index = pickle.load(file)
    
    for image_path in images:
        if image_path not in old_index:
            preprocessed_image = preprocess_image(image_path).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(preprocessed_image)
            index[image_path] = image_features
        else: 
            index[image_path] = old_index[image_path]
    
    # Save data to file so it doesn't need to be processed again
    with open('data.pkl', 'wb') as file:
        pickle.dump(index, file)
    return index

# Function to search for images based on a given prompt
def search_for_image(prompt, index, clip_model, topk):
    """
    Searches for images similar to a given text prompt:
    1. Encodes the text prompt into CLIP feature embeddings.
    2. Calculates similarity scores between the text features and precomputed image features.
    3. Returns the top k similar images.
    """
    text_features = clip_model.encode_text(clip.tokenize([prompt])).to(device)
    similarities = []

    for image_path, image_features in index.items():
        image_features = image_features.to(device)
        similarity = (image_features @ text_features.T).mean().item()  # Check features similarity
        similarities.append((image_path, similarity))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_matches = similarities[:topk]

    return top_matches

# Function to display an image
def display_image(image_path):
    """
    Displays an image given its file path.
    """
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# Main function to orchestrate the program
def main():
    """
    Main function to run the program:
    1. Asks the user if they want to update the image database.
    2. Creates or loads the image index.
    3. Prompts the user for the number of results desired and the search query.
    4. Searches for images based on the user's input.
    5. Displays the search results.
    """
    update = input("Update db?\n")
    if update == "y":
        image_index = create_image_index(image_folder, clip_model) 
    else: 
        with open('data.pkl', 'rb') as file:
            image_index = pickle.load(file)
        
    topk = input("Please enter how many results you want to see: \n")
    search_prompt = input("Please enter what you're searching for: \n")

    search_results = search_for_image(search_prompt, image_index, clip_model, int(topk))
    
    # Display the results
    print("Search results: ")
    for result in search_results:
        image_path = result[0]
        display_image(image_path)

# Entry point of the program
if __name__ == '__main__':
    main()
