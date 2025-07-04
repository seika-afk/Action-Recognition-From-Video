# 🏃‍♂️ Action Recognition using EfficientNet + TensorFlow

This project demonstrates how to build a video classification (action recognition) model using **TensorFlow**, **EfficientNet**, and temporal pooling over video frame sequences. It’s based on a subset of the UCF101 dataset and closely follows TensorFlow’s official video data loading tutorials.

---

## 📁 Dataset

We use a subset of the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php), structured as:

```
UCF101_subset/
├── train/
│   ├── ApplyingMakeup/
│   ├── ApplyingLipstick/
│   └── ...
└── val/
    ├── ApplyingMakeup/
    ├── ApplyingLipstick/
    └── ...
```

Each subfolder contains `.avi` videos for a specific action class.

---

## 🧠 Workflow Overview

### 1. Frame Extraction
We use `frames_from_video_file()` to:
- Open the video using OpenCV
- Sample 10 evenly spaced frames per clip

### 2. FrameGenerator
A Python generator class that:
- Iterates over folders and collects video paths + class labels
- Extracts frames and yields `(frames, label)` tuples
- Compatible with `tf.data.Dataset.from_generator`

### 3. Dataset Creation
```python
train_ds = tf.data.Dataset.from_generator(FrameGenerator(...), output_signature=...)
val_ds = tf.data.Dataset.from_generator(FrameGenerator(...), output_signature=...)
```
Each item:
- `frames`: shape `(10, 224, 224, 3)` (10 RGB frames)
- `label`: scalar integer (class ID)

---

## 🏗️ Model Architecture

Using `EfficientNetB0` as a feature extractor on each frame (via `TimeDistributed`):

```python
model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.TimeDistributed(tf.keras.applications.EfficientNetB0(include_top=False)),
  tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D()),
  tf.keras.layers.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(num_classes)  # 10 for UCF101 subset
])
```

- `Rescaling`: Normalizes pixels
- `TimeDistributed(EfficientNetB0)`: Processes each frame individually
- `GlobalAveragePooling1D`: Aggregates across time (frames)
- `Dense`: Outputs logits for classification

---

## 🔧 Training

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss')]
)
```

---

## 🧪 Inference on a Single Video

To classify a new video:

```python
frames = frames_from_video_file("video_test.avi", n_frames=10)
frames = tf.expand_dims(frames, axis=0)  # Add batch dimension
pred = model(frames)
label = tf.argmax(pred, axis=-1).numpy()[0]
print("Predicted class:", label)
```

---

## 📦 Requirements

- Python ≥ 3.8  
- TensorFlow ≥ 2.11  
- OpenCV (`cv2`)  
- tqdm  
- numpy  
- pathlib  

---

## 🙋‍♂️ Credits

- Inspired by TensorFlow tutorials:
  - [Load video data](https://www.tensorflow.org/tutorials/load_data/video)

- Dataset: UCF101 from UCF Center for Research in Computer Vision
