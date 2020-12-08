# AWE-W segmentation

Solution of HW 2 of the Image based biometry course from University of Ljubljana (winter 2020).

Implements a simple U-Net to segment ears from headshot photos of people. Uses the AWE-W dataset.

See the `.pdf` report for details.

## Running the model

See `requirements.txt` for required dependencies.

You need to place the AWE-W dataset in folder `AWEForSegmentation` it should contain four subfolders: train, trainannot, test and testannot. That is, the folder structure is exactly the same as the homowork zip.

Script `awe_ear_segmentation` can be run directly and must be in the same directory as `AWEForSegmentation` data folder. By default the training part is commented out and pretrained weights are loaded. Feel free to uncomment the code and retrain the model.
