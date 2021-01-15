# AWE-W Ear dataset for biometric tasks

Solution of [HW 2](#hw-2---segmentation) and [HW 3](#hw-3---recognition) of the Image based biometry course from University of Ljubljana (winter 2020).

See `requirements.txt` for required dependencies, note that the scripts were run with Python 3.7.6. The scripts can also be run in Python Interactive to see additional markdown styling.

## HW 2 - segmentation

Implements a simple U-Net to segment ears from headshot photos of people. See the `segmentation.pdf` report for details.

### Running the model

You need to place the AWE-W dataset in folder `AWEForSegmentation` it should contain four subfolders: train, trainannot, test and testannot. That is, the folder structure is exactly the same as the homowork zip.

Script `awe_ear_segmentation.py` can be run directly and must be in the same directory as `AWEForSegmentation` data folder. By default the training part is commented out and pretrained weights are loaded. Feel free to uncomment the code and retrain the model.

To get the pretrained weights download this [folder](https://drive.google.com/drive/folders/1gQtIAd3tgV1k3ASHhOnFCidKtPbq-ySJ?usp=sharing) and put it in the same directory as script.

## HW 3 - recognition

Implements a classification model for ear recognition. The already precropped ear images from the dataset are used. Alternatively, cropping based on the output of segmentation done in HW 2 could be easily implemented to obtain the full biometric pipeline based on ear modality. See the `recognition.pdf` report for details.

### Running the model

You need to place the AWE-W cropped dataset in folder `awe` it should contain 100 subfolders, one for each subject. The paths and train/test split is done based on the `awe-translation.csv` file.

Script `awe_ear_recognition.py` can be run directly and must be in the same directory as `awe` data folder and the translation file. By default the training part is commented out and pretrained weights are loaded. Feel free to uncomment the code and retrain the model.
