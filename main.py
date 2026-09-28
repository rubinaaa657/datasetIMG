import os, json, random
import numpy as np
import pandas as pd
from PIL import Image


BASE_DIR = "dataset"
IMAGES_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

np.random.seed(24)
random.seed(24)

TOTAL_IMAGES = 1000
SIZE = 64

classes = ["dark", "grey", "light", "white"]
base_brightness = {
    "dark": 50,
    "grey": 120,
    "light": 190,
    "white": 240
}

images_paths = []
labels = []
features = []
json_metadata = []

for i in range(TOTAL_IMAGES):
    klass = np.random.choice(classes)
    base_val = base_brightness[klass]
    
    shum = np.random.randint(-25, 25)
    
    final_brightness = max(0, min(255, base_val + shum))
    
    img = Image.new('L', (SIZE, SIZE), color=final_brightness)
    
    filename = f"{klass}_{i}.png"
    local_path = os.path.join(IMAGES_DIR, filename)
    
    img.save(local_path)
    
    git_path = f"images/{filename}"
    normalized_brightness = final_brightness / 255.0
    
    images_paths.append(git_path)
    labels.append(klass)
    features.append(normalized_brightness)
    
    json_metadata.append({
        "id": i,
        "image_path": git_path,
        "label": klass,
        "attributes": {
            "brightness": final_brightness,
            "normalized_brightness": round(normalized_brightness, 4),
            "width": SIZE,
            "height": SIZE
        }
    })

df = pd.DataFrame({
    "path": images_paths,
    "label": labels,
    "brightness": features
})
df.to_csv(os.path.join(BASE_DIR, "dataset.csv"), index=False, encoding="utf-8")

with open(os.path.join(BASE_DIR, "dataset.json"), "w", encoding="utf-8") as f:
    json.dump(json_metadata, f, ensure_ascii=False, indent=2)

target = df["label"]
features_df = df[["brightness"]]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    features_df, target, test_size=0.2, random_state=24
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=24
)

print(f"{TOTAL_IMAGES} black and white pics generated")
print(f"saved '{BASE_DIR}/dataset.csv' and '{BASE_DIR}/dataset.json'")
print(f"size – {X_train.shape}, "
      f"valid – {X_val.shape}, text – {X_test.shape}")
