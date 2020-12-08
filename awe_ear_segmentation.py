# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib

# %%
BASE_PATH = "./AWEForSegmentation"

# %%
TRAIN_DATA_FOLDER = pathlib.Path(BASE_PATH + "/train")
TEST_DATA_FOLDER = pathlib.Path(BASE_PATH + "/test")

TRAIN_MASK_FOLDER = pathlib.Path(BASE_PATH + "/trainannot")
TEST_MASK_FOLDER = pathlib.Path(BASE_PATH + "/testannot")


# %% [markdown]
# ## Settings

# %%
# Choose dimensions that are multiple of 64 to utilize full power of U-Net
IMAGE_WIDTH = 448
IMAGE_HEIGHT = 320

# %%
CLASSES = 2


# %% [markdown]
# ## Load data

# %%
def load_filenames(data_dir):
    image_count = len(list(data_dir.glob('*.png')))
    print(f"Prefetched: {image_count} images")
    test_filenames = tf.data.Dataset.list_files(str(data_dir/'*'), shuffle=False)
    return image_count, test_filenames.shuffle(image_count, reshuffle_each_iteration=False)

# %%
TEST_SIZE, test_data = load_filenames(TEST_DATA_FOLDER)

# %%
TRAIN_SIZE, train_data = load_filenames(TRAIN_DATA_FOLDER)


# %% [markdown]
# ## Create `image, label` tuples

# %%
def get_mask(image_path):
    # Produce correct path.
    mask_path = tf.strings.regex_replace(image_path, '(test|train)', '\\1annot')
    print(mask_path)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    return tf.image.resize(mask, [IMAGE_HEIGHT, IMAGE_WIDTH])

# %%
def get_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    return tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])  # Necessary for meaningful batching.

# %%
def prepare_example(image_path):
    image = get_image(image_path)
    mask = get_mask(image_path)
    return tf.image.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH)), tf.image.resize(mask, (IMAGE_HEIGHT, IMAGE_WIDTH))


# %%
# Process datasets.
train_data = train_data.map(prepare_example, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)
test_data = test_data.map(prepare_example, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)


# %% [markdown]
# ## Create validation split

# %%
VAL_RATIO = 0.85

# %%
train_indices = np.random.choice(range(TRAIN_SIZE), int(VAL_RATIO * TRAIN_SIZE), replace=False)
print(f"Train size: {train_indices.shape[0]}")

# %%
val_indices = list(set(range(TRAIN_SIZE)) - set(train_indices))
print(f"Validation size: {len(val_indices)}")

# %%
def subset_dataset(dataset, indices):
    return dataset.enumerate().filter(lambda i, t: tf.reduce_any(i == indices)).map(lambda j, u: u)

# %%
train_data = subset_dataset(train_data, train_indices)
val_data = subset_dataset(train_data, val_indices)


# %% [markdown]
# ## Image augmentations

# %%
HFLIP = 0.5
HSV = [0.3, 0.3, 0.3]
GRAYSCALE = 0.2

# %%
def hflip(image, mask):
    if tf.random.uniform([]) < HFLIP:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    return image, mask

def hsv(image, mask):
    if tf.random.uniform([]) < HSV[0]:
        image = tf.image.adjust_hue(image, tf.random.uniform([], -1, 1))
    if tf.random.uniform([]) < HSV[1]:
        image = tf.image.adjust_saturation(image, tf.random.uniform([], 0.2, 5))
    if tf.random.uniform([]) < HSV[2]:
        image = tf.image.adjust_brightness(image, tf.random.uniform([], 0, 1))
    return image, mask

def grayscale(image, mask):
    if tf.random.uniform([]) < GRAYSCALE:
        image = tf.where(mask == 1, tf.image.rgb_to_grayscale(image), image)
    return image, mask

# %%
def augment_image(image, mask):
    return hflip(*hsv(*grayscale(image, mask)))

# %%
# Augment validation and train data separately to not leak infomartion into evaluation.
train_data = train_data.concatenate(train_data.map(augment_image))
val_data = val_data.concatenate(val_data.map(augment_image))


# %% [markdown]
# ## Batching and caching

# %%
SHUFFLE_BUFFER = 1000
BATCH_SIZE = 32

# %%
def cache_oneoff_tasks(dataset):
  dataset = dataset.cache()
  dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER)
  dataset = dataset.batch(BATCH_SIZE)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset

train_data = cache_oneoff_tasks(train_data)
val_data = cache_oneoff_tasks(val_data) # No need to shuffle, but it's so small it does not matter.


# %% [markdown]
# ## Visualization callback

# %%
# Visualization callback to see how mask prediction progresses.
class MaskCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        plot_mask_predictions(val_data, 3)

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

# %%
def plot_mask_predictions(dataset, examples=3):
    plt.figure(figsize=(21, 7 * examples))

    for i, (image, mask) in enumerate(dataset.take(examples)):
        ax = plt.subplot(examples, 3, 3*i+1)
        plt.imshow(image[0].numpy().astype("uint8"))
        plt.axis("off")
            
        ax = plt.subplot(examples, 3, 3*i+2)
        plt.imshow(mask[0].numpy().astype("uint8").squeeze(axis=2), cmap='gray', vmin=0, vmax=1)
        plt.axis("off")


        pred_mask = create_mask(model.predict(image))

        ax = plt.subplot(examples, 3, 3*i+3)
        plt.imshow(pred_mask.numpy().astype("uint8").squeeze(axis=2), cmap='gray', vmin=0, vmax=1)
        plt.axis("off")

    plt.show()


# %% [markdown]
# ## Load Efficient Net (smallest B0)

# %%
efficient_net_b0 = tf.keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_tensor=None, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))

# %%
encoder_layers = [
    'block1a_project_bn',
    'block2b_add',
    'block3b_add',
    'block5c_add',
    'top_activation'    # \2^5 (12x15 for 360x480)
]

encoder = [efficient_net_b0.get_layer(l).output for l in encoder_layers]
encoder = tf.keras.Model(inputs=efficient_net_b0.input, outputs=encoder)

# %%
# Don't train 'imagenet' weights.
encoder.trainable = False


# %% [markdown]
# ## Segmentation head

# %%
# U-net decoder.
def upsampling_layer(inputs, filters, kernel=(3,3), strides=(2,2), padding='same'):
    """
    General Conv2DTranspose -> BN -> (Dropout) -> ReLU layer.
    """
    x = tf.keras.layers.Conv2DTranspose(filters, kernel, strides=strides, padding=padding, use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

# %%
decoder_filters = [512, 256, 128, 64, 32]  # Number of filters as we progress down the decoder.

# %%
def calculate_iou(y_true, y_pred):
    """
    Returns IoU of two binary masks. True mask must have 1 channel with class 
    and prediction mask 2 channels with probabilities / logits.
    """
    y_true_mask = tf.reshape(tf.math.round(y_true) == 1, [-1, IMAGE_HEIGHT * IMAGE_WIDTH])
    y_pred = tf.math.argmax(y_pred, axis=-1)
    y_pred_mask = tf.reshape(y_pred == 1, [-1, IMAGE_HEIGHT * IMAGE_WIDTH])

    intersection_mask = tf.math.logical_and(y_true_mask, y_pred_mask)
    union_mask = tf.math.logical_or(y_true_mask, y_pred_mask)

    intersection = tf.reduce_sum(tf.cast(intersection_mask, tf.float32), axis=1)
    union = tf.reduce_sum(tf.cast(union_mask, tf.float32), axis=1)

    iou = tf.where(union == 0, 1., intersection / union)
    return iou

class IntersectionOverUnion(tf.metrics.Mean):
    def update_state(self, y_true, y_pred, sample_weight=None):
        iou = calculate_iou(y_true, y_pred)
        return super().update_state(iou, sample_weight)

# %%
inputs = tf.keras.layers.Input(shape=[IMAGE_HEIGHT, IMAGE_WIDTH, 3])

# Encoding.
# Set `training` to false to keep BN in inference mode for fine tuning.
encoder_outputs = encoder(inputs, training=False)

# Last model layer output.
x = encoder_outputs[-1]
# Reverse since we go from smallest to largest.
encoder_outputs = reversed(encoder_outputs[:-1])

# Forming U-net.
for encoder_output, filters in zip(encoder_outputs, decoder_filters):
    decoder_output = upsampling_layer(x, filters)
    x = tf.keras.layers.Concatenate()([decoder_output, encoder_output]) # Skip connection.

# Get mask prediction back to original size.
outputs = tf.keras.layers.Conv2DTranspose(CLASSES, 3, strides=2, padding='same')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

# %%
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        'accuracy',
        IntersectionOverUnion(name='IoU'),
    ])


# %% [markdown]
# ## Model training

# %%
# Remember that it has to be explicitly set in checkpoints.
EXP_ID = 1

# %%
EPOCHS = 15
MODEL_NAME = f"/{EXP_ID}_run_{EPOCHS}e_{BATCH_SIZE}b"

# %%
# Save all checkpoints.
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoints/run_1/cp-{epoch:04d}.ckpt", 
    verbose=1, 
    save_weights_only=True)

# %%
# Uncomment all code below to train the model.
# ============================================
# model_history = model.fit(train_data, epochs=EPOCHS, validation_data=val_data, callbacks=[MaskCallback(), checkpoint_callback])


# # %% [markdown]
# # ## Visualize loss

# # %%
# loss = model_history.history['loss']
# val_loss = model_history.history['val_loss']

# epochs = range(EPOCHS)

# plt.figure()
# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'go', label='Validation loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim([0, 0.1])
# plt.legend()
# plt.show()


# # %% [markdown]
# # ## Finetuning

# # %%
# FINETUNE_FROM = 180
# FINE_EPOCHS = 10
# # Enable Efficient net training, but only from around 2/3 of layers.
# encoder.trainable = True
# for layer in encoder.layers[:FINETUNE_FROM]:
#   layer.trainable =  False

# # %%
# # Recompile model to take effect.
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Learn at lower rate (10x).
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[
#         'accuracy',
#         IntersectionOverUnion(name='IoU'),
#     ])

# # %%
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath="checkpoints/fine_1/cp-{epoch:04d}.ckpt", 
#     verbose=1, 
#     save_weights_only=True)

# # %%
# model_history_fine = model.fit(train_data, epochs=EPOCHS+FINE_EPOCHS, initial_epoch=model_history.epoch[-1], validation_data=val_data, callbacks=[MaskCallback(), checkpoint_callback])


# # %% [markdown]
# # ## Visualize loss

# # %%
# loss += model_history_fine.history['loss']
# val_loss += model_history_fine.history['val_loss']

# # %%
# epochs = range(EPOCHS + FINE_EPOCHS + 1)

# plt.figure()
# plt.plot(epochs, loss, 'r', label='Training loss')
# plt.plot(epochs, val_loss, 'go', label='Validation loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.text(EPOCHS + 0.3,0.0075,'Finetuning start',rotation=90)

# plt.annotate('Validation loss flat', xy=(20, 0.0025), xytext=(18, 0.005),
#             arrowprops=dict(arrowstyle="->", facecolor='k'),
#             )

# plt.ylim([0, 0.02])
# plt.vlines(x=EPOCHS, ymin=0, ymax=0.02)
# plt.legend()
# plt.show()


# %% [markdown]
# ## Results

# %%
# Load weights from the selected epoch.
model.load_weights("checkpoints/cp-0020.ckpt")

# %%
# See metrics on test data.
model.evaluate(test_data.batch(BATCH_SIZE))

# %%
# Predict test masks.
test_masks = model.predict(test_data.map(lambda i, m: tf.expand_dims(i, 0)))

# %%
# See distribution of IoU in test data.
iou_test = []

for y_pred, y_true in zip(test_masks, test_data.map(lambda i, m: m)):
    iou = calculate_iou(y_true, y_pred)
    iou_test.append(iou)

plt.figure()
a = plt.plot(range(len(test_data)), sorted(iou_test))
plt.show()

# %%
# See several mask predictions.
plot_mask_predictions(test_data.batch(1), 10)

# %%
