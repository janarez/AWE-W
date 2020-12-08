# AWE-W segmentation

Solution of HW 2 of the Image based biometry course from University of Ljubljana (winter 2020).

Implements a simple U-Net to segment ears from headshot photos of people. Uses the AWE-W dataset.

See the `.pdf` report for details.

## Running the model

See `requirements.txt` for required dependencies, note that the script was run with Python 3.7.6.

You need to place the AWE-W dataset in folder `AWEForSegmentation` it should contain four subfolders: train, trainannot, test and testannot. That is, the folder structure is exactly the same as the homowork zip.

Script `awe_ear_segmentation.py` can be run directly and must be in the same directory as `AWEForSegmentation` data folder. By default the training part is commented out and pretrained weights are loaded. Feel free to uncomment the code and retrain the model.

To get the pretrained weights download this [folder](https://drive.google.com/drive/folders/1gQtIAd3tgV1k3ASHhOnFCidKtPbq-ySJ?usp=sharing) and put it in the same directory as script.

The script can also be run in Python Interactive to see additional markdown styling.
