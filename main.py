import os, json, random
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

base_dir = "dataset"
images_dir = os.path.join(base_dir, "images")

os.makedirs(images_dir, exist_ok=True)

np.random.seed(24)
random.seed(24)
tf.random.set_seed(24)

total_images = 1000
size = 64

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

for i in range(total_images):
    klass = np.random.choice(classes)
    base_val = base_brightness[klass]
    shum = np.random.randint(-25, 25)
    final_brightness = max(0, min(255, base_val + shum))
    img = Image.new("L", (size, size), color=final_brightness)
    filename = f"{klass}_{i}.png"
    local_path = os.path.join(images_dir, filename)
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
            "width": size,
            "height": size
        }
    })

df = pd.DataFrame({
    "path": images_paths,
    "label": labels,
    "brightness": features
})
df.to_csv(os.path.join(base_dir, "dataset.csv"), index=False, encoding="utf-8")

with open(os.path.join(base_dir, "dataset.json"), "w", encoding="utf-8") as f:
    json.dump(json_metadata, f, ensure_ascii=False, indent=2)

target = df["label"]
features_df = df[["brightness"]]

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    features_df, target_encoded, test_size=0.2, random_state=24
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=24
)

print(f"{total_images} black and white pics generated")
print(f"saved '{base_dir}/dataset.csv' and '{base_dir}/dataset.json'")
print(f"size – {X_train.shape}, valid – {X_val.shape}, text – {X_test.shape}")

model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(4, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

base_loss, base_acc = model.evaluate(X_test, y_test, verbose=0)

new_features = []
new_labels = []

for i in range(200):
    klass = np.random.choice(classes)
    base_val = base_brightness[klass]
    shum = np.random.randint(-25, 25)
    final_brightness = max(0, min(255, base_val + shum))
    new_features.append(final_brightness / 255.0)
    new_labels.append(klass)

X_new = pd.DataFrame({"brightness": new_features})
y_new_encoded = label_encoder.transform(new_labels)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.fit(
    X_new, y_new_encoded,
    epochs=5,
    batch_size=16,
    verbose=1
)

final_loss, final_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"base accuracy: {base_acc:.4f}")
print(f"final accuracy: {final_acc:.4f}")

if final_acc > base_acc:
    print("fine-tuning worked: accuracy increased!")
elif final_acc == base_acc:
    print("accuracy stayed the same.")
else:
    print("accuracy dropped: catastrophic forgetting occurred.")