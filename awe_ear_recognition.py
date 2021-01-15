# %%
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.integrate import simps

from matplotlib import pyplot as plt

# %%
DATA_FOLDER = '../awe'
META_FILE = '../awe-translation.csv'

# %%
meta_file = pd.read_csv(META_FILE)

# %%
# Subtract one from ID to start labelling at zero.
meta_file['Subject ID'] -= 1
CLASSES = len(meta_file['Subject ID'].unique())

# %%
test_images = meta_file[meta_file['AWE-Full image path'].apply(lambda x: x.split('/')[0] == 'test')]['AWE image path'].values
train_images = meta_file[meta_file['AWE-Full image path'].apply(lambda x: x.split('/')[0] == 'train')]['AWE image path'].values

test_images_labels = meta_file[meta_file['AWE-Full image path'].apply(lambda x: x.split('/')[0] == 'test')]['Subject ID'].values
train_images_labels = meta_file[meta_file['AWE-Full image path'].apply(lambda x: x.split('/')[0] == 'train')]['Subject ID'].values

print(f'Train size: {train_images.shape[0]}')
print(f'Test size: {test_images.shape[0]}')

# %%
training_data = tf.data.Dataset.from_tensor_slices((train_images, train_images_labels))
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_images_labels))

# %%
# Analyze image sizes.

# import pathlib
# import PIL

# data = pathlib.Path(DATA_FOLDER)

# max_height = 0
# max_width = 0

# for path in data.glob('*/*.png'):
#     image = PIL.Image.open(path)
#     image_array = np.array(image)

#     h, w = image_array.shape[:2]
#     if h > max_height:
#         max_height = h
#     if w > max_width:
#         max_width = w

# %%
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3

def get_image(image_path, label):
    image_path = tf.strings.join([DATA_FOLDER, image_path], separator='/')
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    return image, label

# %%
training_data = training_data.map(get_image).cache()
test_data = test_data.map(get_image)

# %%
# Original random validation split. Forewent, because of limited data.

# VAL_RATIO = 0.75

# train_indices = np.random.choice(range(train_images.shape[0]), int(VAL_RATIO * train_images.shape[0]), replace=False)
# print(f"Train size: {train_indices.shape[0]}")
# val_indices = list(set(range(train_images.shape[0])) - set(train_indices))
# print(f"Validation size: {len(val_indices)}")

# def subset_dataset(dataset, indices):
#     return dataset.enumerate().filter(lambda i, t: tf.reduce_any(i == indices)).map(lambda j, u: u)

# train_data = subset_dataset(train_data, train_indices)
# val_data = subset_dataset(train_data, val_indices)

# %%
# Augmentations.
HSV = [0.3, 0.3, 0.3]

def random_hsv(image):
    if tf.random.uniform([]) < HSV[0]:
        image = tf.image.adjust_hue(image, tf.random.uniform([], -0.1, 0.1))
    if tf.random.uniform([]) < HSV[1]:
        image = tf.image.adjust_saturation(image, tf.random.uniform([], 0.25, 4))
    if tf.random.uniform([]) < HSV[2]:
        image = tf.image.adjust_brightness(image, tf.random.uniform([], -1, 1))
    return image

HFLIP_PROB = 0.5
HSV_PROB = 0.5
GRAY_PROB = 0.5
SHIFT = 24

def image_augment(image, label):
    # Horizontal flip.
    if tf.random.uniform([]) < HFLIP_PROB:
        image = tf.image.flip_left_right(image)

    # Color augmentations.
    if tf.random.uniform([]) < HSV_PROB:
        image = random_hsv(image)

    # Grayscale.
    if tf.random.uniform([]) < GRAY_PROB:
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)

    # Shifted resize to desired dimensions.
    shifted_height = tf.random.uniform([], minval=IMAGE_HEIGHT, maxval=IMAGE_HEIGHT + SHIFT, dtype=tf.int32)
    shifted_width = tf.random.uniform([], minval=IMAGE_WIDTH, maxval=IMAGE_WIDTH + SHIFT, dtype=tf.int32)

    image = tf.image.resize(image, [shifted_height, shifted_width])
    image = tf.image.random_crop(image, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

    return image, label

def resize_image(image, label):
    image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return image, label

# %%
SHUFFLE_BUFFER = 1000
BATCH_SIZE = 32

def split_batch_and_tune(dataset, skip_size, take_size, augment=False):
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER)
    dataset = dataset.skip(skip_size).take(take_size)
    if augment:
        dataset = dataset.map(image_augment)
    else:
        dataset = dataset.map(resize_image)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Augment only training dataset.
train_data = split_batch_and_tune(training_data, 0, 500, augment=True)

# Resize the rest.
val_data = split_batch_and_tune(training_data, 500, 250)
test_data = test_data.map(resize_image).batch(BATCH_SIZE).cache()

# %% [markdown]
# ## Within model augmentations
# We end up not using these positional augmentations and rather use the shifts
# implemented above.

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(-0.2, 0.2), fill_mode="reflect", interpolation='nearest'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.15),
    tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
])

# %%
L2 = tf.keras.regularizers.l2(0.0001)

# %%
x = inputs = tf.keras.layers.Input(shape=[IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

# Augmentations.
#x = data_augmentation(x)

efficient_net = tf.keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_tensor=x)
efficient_net.trainable = False

x = efficient_net.output

x = tf.keras.layers.Conv2D(32, 3, strides=1, padding='same', kernel_regularizer=L2)(x)
x = tf.keras.layers.MaxPool2D(2)(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.Conv2D(64, 3, strides=1, padding='same', kernel_regularizer=L2)(x)
x = tf.keras.layers.MaxPool2D(2)(x)
x = tf.keras.layers.BatchNormalization()(x)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)

outputs = tf.keras.layers.Dense(CLASSES, activation=tf.nn.softmax)(x)

# %%
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# %%
model.compile(
    optimizer=tf.optimizers.Adam(0.01),
    loss=tf.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.metrics.SparseCategoricalAccuracy(name='accuracy'),
    ],
)

# %%
EPOCHS = 100

# %%
def scheduler(epoch, lr):
    if epoch < 25:
        return 0.01
    return 0.001

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

# %%
# Save all checkpoints.
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/final-yes/cp-{epoch:04d}.ckpt",
    save_weights_only=True
)

# %%
model_history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[lr_callback, checkpoint_callback]
)

# %% [markdown]
# Run again for nonaugmented version. Set `augment=False` in these lines:
# ```python
# # Augment only training dataset.
# train_data = split_batch_and_tune(training_data, 0, 500, augment=True)
# ```

model_history_no = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=[lr_callback, checkpoint_callback]
)

# %%
model.evaluate(test_data)

# %%
# Loss
loss = model_history.history['loss'].copy()
val_loss = model_history.history['val_loss'].copy()

plt.figure()
plt.plot(range(EPOCHS), loss, 'r', label='Training loss')
plt.plot(range(EPOCHS), val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.legend()
plt.show()

# %%
# Accuracies
accuracy = model_history.history['accuracy'].copy()
no_accuracy = model_history_no.history['accuracy'].copy()

plt.figure()
plt.plot(range(EPOCHS), accuracy, 'g', label='w\ augmentations')
plt.plot(range(EPOCHS), no_accuracy, 'r', label='w\o augmentations')
+50
plt.xlabel('Epoch')
plt.ylabel('Rank-1')
plt.ylim([0, 1])
plt.legend()
plt.show()

# %%
model.load_weights("checkpoints/yes/cp-0100.ckpt")
# model.load_weights("checkpoints/no/cp-0100.ckpt")

# %%
# CMC Curve
top_k_accuracies = []
for i in range(CLASSES):
    top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=i+1)
    top_k_accuracies.append(top_k)

for test_images, test_labels in test_data:
    pred_labels = model(test_images, training=False)

    for i in range(CLASSES):
        top_k_accuracies[i].update_state(test_labels, pred_labels)

# %%
cmc_data = [top_k.result().numpy() for top_k in top_k_accuracies]

# %%
# Single value metrics.
area = simps(cmc_data, dx=1)
print(f"AUCMC: {area}")
print(f'Rank-1: {cmc_data[0]}')
print(f'Rank-5: {cmc_data[4]}')

# %%
plt.figure()
plt.plot(range(1,1+CLASSES), cmc_data, 'g', label='w\ augmentations')
#plt.plot(range(1,1+CLASSES), cmc_data_no, 'r', label='w\o augmentations')
plt.xticks([1] + list(range(10, 101, 10)))
plt.xlabel('Rank')
plt.ylabel('Recognition Rate')
plt.legend()
plt.show()

# %%
# Misclassification per person.
person_miss = np.zeros(CLASSES)
person_miss_5 = np.zeros(CLASSES)
person_total = np.zeros(CLASSES)

for test_images, test_labels in test_data:
    pred_labels = model(test_images, training=False)

    for y_true, y_pred in zip(test_labels, pred_labels):
        y_top = np.argsort(y_pred)
        if y_true not in y_top[-1:]:
            person_miss[y_true] += 1
        if y_true not in y_top[-5:]:
            person_miss_5[y_true] += 1
        person_total[y_true] += 1

# %%
plt.figure()
ax = plt.gca()
counts, _, patches = plt.hist(person_miss_5 / person_total, 100, density=True, facecolor='r', alpha=0.5, label='Rank-5')
for count, patch in zip(counts,patches):
    if count != 0:
        ax.annotate(str(int(count)), xy=(patch.get_x()+0.02, patch.get_height()))

counts, _, patches = plt.hist(person_miss / person_total, 100, density=True, facecolor='g', alpha=0.5, label='Rank-1')
for count, patch in zip(counts,patches):
    if count != 0:
        ax.annotate(str(int(count)), xy=(patch.get_x()+0.02, patch.get_height()))

plt.xlabel('Misclassification rate')
plt.ylabel('Number of subjects')
plt.legend()
plt.legend()
plt.show()
