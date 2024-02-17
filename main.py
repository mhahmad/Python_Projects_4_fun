import os
import glob
import torch
from PIL import Image
import torchvision.transforms as transforms
import clip
import matplotlib.pyplot as plt
import pickle

image_folder = "./images"
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("RN50", device=device)


def preprocess_image(image_path):
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),#0.5 common for RGB images
    ])
    
    image = preprocess(image).unsqueeze(0)
    return image


def create_image_index(image_folder, clip_model):#creates a dictionary of (image_path:image_features)
    index = {}#dictionary for (image_path:image_features)
    images = glob.glob(os.path.join(image_folder, "*.jpg"))+glob.glob(os.path.join(image_folder, "*.jpeg"))+glob.glob(os.path.join(image_folder, "*.png"))
    with open('data.pkl', 'rb') as file:
        old_index = {}
    
    for image_path in images:
        if image_path not in old_index:
            preprocessed_image = preprocess_image(image_path).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(preprocessed_image)
            index[image_path] = image_features
        else: 
            index[image_path]=old_index[image_path]
        
    
    #save data to file so you don't have to process them again
    with open('data.pkl', 'wb') as file:
        pickle.dump(index,file)
    return index


def search_for_image(prompt, index, clip_model, topk):
    text_features = clip_model.encode_text(clip.tokenize([prompt])).to(device)
    similarities = []

    for image_path, image_features in index.items():
        image_features = image_features.to(device)
        similarity = (image_features @ text_features.T).mean().item()#check features similarity
        similarities.append((image_path, similarity))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    top_matches = similarities[:topk]

    return top_matches




def display_image(image_path):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


    
def main():
    update=input("Update db?\n")
    if update=="y":
        image_index = create_image_index(image_folder, clip_model) 
    else: 
        with open('data.pkl', 'rb') as file:
            image_index = pickle.load(file)
        
    topk = input("please enter how many results you want to see \n")
    search_prompt = input("please enter what you're searching for: \n")

    search_results = search_for_image(search_prompt, image_index, clip_model,int(topk))
    
 
    # Display the results
    print("Search results: ")
    for result in search_results:
        image_path = result[0]
        display_image(image_path)

if __name__ == '__main__':
    main()

