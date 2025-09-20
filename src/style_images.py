import os
import numpy as np
import tensorflow as tf

STYLE_IMAGES = {
    "starry_night": "assets/styles/starry_night.jpg",
    "mona_lisa": "assets/styles/mona_lisa.jpg",
    "cubism": "assets/styles/cubism.jpg",
    "impressionism": "assets/styles/impressionism.jpg",
    "surrealism": "assets/styles/surrealism.jpg",
    "cyberpunk": "assets/styles/cyberpunk.jpg"
}

STYLE_FEATURES_DIR = "style_features"
os.makedirs(STYLE_FEATURES_DIR, exist_ok=True)

def save_style_features():
    model = get_vgg_model()  # Load VGG19 model
    
    for style_name, img_path in STYLE_IMAGES.items():
        image = load_img(img_path)
        features = extract_style_features(image, model)
        feature_path = os.path.join(STYLE_FEATURES_DIR, f"{style_name}.npy")
        np.save(feature_path, features)  # Save features as a NumPy file
        print(f"Saved features for {style_name}")

save_style_features()
