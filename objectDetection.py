import numpy as np
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import kagglehub

# Download the Fruits Detection dataset from Kaggle
path = kagglehub.dataset_download("lakshaytyagi01/fruit-detection")

print("Path to Fruits Detection dataset files:", path)



dataset_path = "/root/.cache/kagglehub/datasets/lakshaytyagi01/fruit-detection/versions/1"

# List full directory tree in dataset_path
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files[:5]:  # limit to 5 files per folder
        print(f"{subindent}{f}")




base_path = "/root/.cache/kagglehub/datasets/lakshaytyagi01/fruit-detection/versions/1/Fruits-detection"

valid_path = os.path.join(base_path, "valid")
print("Valid folder contents:", os.listdir(valid_path))

print("Valid Images:", os.listdir(os.path.join(valid_path, "images"))[:5])
print("Valid Labels:", os.listdir(os.path.join(valid_path, "labels"))[:5])



im_width, im_height = 75, 75
NUM_CLASSES = 6
CLASS_NAMES = ['Apple', 'Banana', 'Grapes', 'Orange', 'Pineapple', 'Watermelon']

# ---------- Drawing Utilities ----------
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red', thickness=2, display_str=None, use_normalized_coordinates=True):
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        left, right = xmin * im_width, xmax * im_width
        top, bottom = ymin * im_height, ymax * im_height
    else:
        left, right, top, bottom = xmin, xmax, ymin, ymax

    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

    if display_str:
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), display_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        margin = int(0.05 * text_height)
        draw.rectangle([(left, top - text_height - 2*margin), (left + text_width + 2*margin, top)], fill=color)
        draw.text((left + margin, top - text_height - margin), display_str, fill='white', font=font)

def draw_bounding_boxes_on_image_array(image, boxes, colors, display_strs):
    image_pil = PIL.Image.fromarray(np.uint8(image))
    for box, color, disp_str in zip(boxes, colors, display_strs):
        draw_bounding_box_on_image(image_pil, box[0], box[1], box[2], box[3], color=color, display_str=disp_str)
    return np.array(image_pil)



# ---------- Dataset Loader ----------
def yolo_to_bbox(yolo_line):
    cls, cx, cy, w, h = map(float, yolo_line.strip().split())
    xmin = max(0, cx - w/2)
    ymin = max(0, cy - h/2)
    xmax = min(1, cx + w/2)
    ymax = min(1, cy + h/2)
    return int(cls), [ymin, xmin, ymax, xmax]

def load_dataset(split_dir, max_samples=None):
    image_dir = os.path.join(split_dir, "images")
    label_dir = os.path.join(split_dir, "labels")
    files = sorted(os.listdir(image_dir))
    if max_samples:
        files = files[:max_samples]

    images, labels, bboxes = [], [], []
    for file in files:
        img_path = os.path.join(image_dir, file)
        img = Image.open(img_path).convert('RGB').resize((im_width, im_height))
        img_arr = np.array(img, dtype=np.float32) / 255.0

        label_path = os.path.join(label_dir, os.path.splitext(file)[0] + ".txt")
        with open(label_path, 'r') as f:
            cls_id, bbox = yolo_to_bbox(f.readline())

        images.append(img_arr)
        labels.append(tf.one_hot(cls_id, NUM_CLASSES).numpy())
        bboxes.append(bbox)

    return np.array(images), np.array(labels), np.array(bboxes)

# ---------- Model Blocks ----------
def feature_extractor(inputs):
    x = tf.keras.layers.Conv2D(16, 3, activation='relu')(inputs)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(x)
    return x

def dense_layers(inputs):
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    return x

def classifier(inputs):
    return tf.keras.layers.Dense(NUM_CLASSES, activation='softmax', name='classification')(inputs)

def bounding_box_regression(inputs):
    return tf.keras.layers.Dense(4, name='bounding_box')(inputs)

def final_model():
    inputs = tf.keras.Input(shape=(im_width, im_height, 3))
    feats = feature_extractor(inputs)
    dense = dense_layers(feats)
    class_out = classifier(dense)
    bbox_out = bounding_box_regression(dense)
    return tf.keras.Model(inputs=inputs, outputs=[class_out, bbox_out])

# ---------- IoU Calculation ----------
def intersection_over_union(pred_boxes, true_boxes):
    ymin_pred, xmin_pred, ymax_pred, xmax_pred = np.split(pred_boxes, 4, axis=1)
    ymin_true, xmin_true, ymax_true, xmax_true = np.split(true_boxes, 4, axis=1)

    inter_ymin = np.maximum(ymin_pred, ymin_true)
    inter_xmin = np.maximum(xmin_pred, xmin_true)
    inter_ymax = np.minimum(ymax_pred, ymax_true)
    inter_xmax = np.minimum(xmax_pred, xmax_true)

    inter_h = np.maximum(inter_ymax - inter_ymin, 0)
    inter_w = np.maximum(inter_xmax - inter_xmin, 0)
    inter_area = inter_h * inter_w

    pred_area = (ymax_pred - ymin_pred) * (xmax_pred - xmin_pred)
    true_area = (ymax_true - ymin_true) * (xmax_true - xmin_true)

    union_area = pred_area + true_area - inter_area + 1e-10
    return inter_area / union_area

# ---------- Plotting ----------
def plot_metrics(history):
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.plot(history.history['classification_loss'], label='Training Loss')
    plt.plot(history.history['val_classification_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Classification Loss')

    plt.subplot(1,3,2)
    plt.plot(history.history['bounding_box_loss'], label='Training Loss')
    plt.plot(history.history['val_bounding_box_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Bounding Box Loss')

    plt.subplot(1,3,3)
    plt.plot(history.history['classification_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_classification_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Classification Accuracy')

    plt.show()

def display_results(images, true_labels, pred_labels, true_boxes, pred_boxes, iou_scores, iou_threshold=0.5):
    n = 10
    idxs = np.random.choice(len(images), n, replace=False)
    images_show = (images[idxs] * 255).astype(np.uint8)
    true_labels = true_labels[idxs]
    pred_labels = pred_labels[idxs]
    true_boxes = true_boxes[idxs]
    pred_boxes = pred_boxes[idxs]
    iou_scores = iou_scores[idxs]

    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        img = draw_bounding_boxes_on_image_array(
    images_show[i],
    boxes=[true_boxes[i], pred_boxes[i]],
    colors=['green', 'red'],
    display_strs=[f"GT: {CLASS_NAMES[np.argmax(true_labels[i])]}",
                  f"Pred: {CLASS_NAMES[pred_labels[i]]}"]
)
        ax.imshow(img)
        color = 'black' if iou_scores[i][0] >= iou_threshold else 'red'
        ax.set_xlabel(f"IOU: {iou_scores[i][0]:.2f}", color=color)
        ax.axis('off')

# ---------- Main ----------
base_path = "/root/.cache/kagglehub/datasets/lakshaytyagi01/fruit-detection/versions/1/Fruits-detection"

train_images, train_labels, train_bboxes = load_dataset(os.path.join(base_path, "train"))
valid_images, valid_labels, valid_bboxes = load_dataset(os.path.join(base_path, "valid"))

model = final_model()
model.summary()

model.compile(
    optimizer='adam',
    loss={'classification': 'categorical_crossentropy', 'bounding_box': 'mse'},
    metrics={'classification': 'accuracy', 'bounding_box': 'mse'}
)

history = model.fit(
    train_images,
    {'classification': train_labels, 'bounding_box': train_bboxes},
    validation_data=(valid_images, {'classification': valid_labels, 'bounding_box': valid_bboxes}),
    epochs=5,
    batch_size=64,
    verbose=1
)

plot_metrics(history)

eval_results = model.evaluate(valid_images, {'classification': valid_labels, 'bounding_box': valid_bboxes}, batch_size=64)
print(f"Evaluation results: {eval_results}")

predictions = model.predict(valid_images)
pred_labels = np.argmax(predictions[0], axis=1)
pred_bboxes = predictions[1]

iou_scores = intersection_over_union(pred_bboxes, valid_bboxes)
print(f"Mean IoU: {np.mean(iou_scores)}")

display_results(valid_images, valid_labels, pred_labels, valid_bboxes, pred_bboxes, iou_scores)
