# AI-Based Automatic Crop Sorting

A complete, local software prototype that learns crop quality categories from labeled images, predicts a category for a new crop image, and recommends a sorting destination. It includes a Streamlit interface and a command-line training workflow.

> This is a vision-only prototype. It does not control a conveyor, camera, or motor. Predictions depend on the images used for training and should be validated before any real-world sorting decisions.

## Features

- Train an image classifier from your own labeled crop photos.
- Upload an image in the web app to see the predicted class, confidence, and destination.
- Configure category names and sorting destinations in `config/sorting_rules.json`.
- Keep datasets and trained model files local; they are excluded from Git by default.

## Requirements

- Python 3.10 or newer
- Labeled images grouped into folders (see **Prepare the dataset**)

## Setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

## Prepare the dataset

Place images in one folder per category. The folder name becomes the model's class label:

```text
data/train/
  Fresh/
    fresh_001.jpg
    fresh_002.jpg
  Unripe/
    unripe_001.jpg
  Damaged/
    damaged_001.jpg
```

Use at least two categories and several varied images per category. Use consistent lighting and backgrounds where possible, and include realistic variation in angle, size, and crop appearance. Keep images representative of the camera and conditions where the model will be used. The project does not include a crop dataset or pretrained model.

## Train

```powershell
python train.py --data-dir data/train
```

The trained model is saved to `artifacts/crop_sorter.joblib` and the class labels to `artifacts/classes.json`. To use a different dataset or model path:

```powershell
python train.py --data-dir path/to/labeled_images --model-out artifacts/crop_sorter.joblib
```

Training fails with a clear message if images are missing, unreadable, or only one class is present.

## Run the app

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, upload a JPG or PNG crop image, and review the prediction. The destination is looked up by predicted class in `config/sorting_rules.json`; unlisted labels use the default route.

## Customize sorting destinations

Edit `config/sorting_rules.json` to match the dataset labels and your process. For example, rename `Fresh`, `Unripe`, and `Damaged` to match your folder names. A destination is guidance displayed in the app; it does not actuate hardware.

## Project layout

```text
app.py                         Streamlit image upload and prediction UI
train.py                       Command-line model training entry point
src/crop_sorter/model.py       Image preprocessing, training, prediction
config/sorting_rules.json      Class-to-destination mapping
data/train/README.md           Dataset folder instructions
artifacts/                     Locally generated model (git-ignored)
```

## Limitations and next steps

- The starter classifier uses HOG image features and a linear SVM. It is intended as an understandable baseline, not a claim of production accuracy.
- Collect balanced, correctly labeled images for each class and assess performance on images kept out of training before relying on results.
- A physical sorter needs additional hardware, camera calibration, real-time inference, actuator control, and safety handling; those are outside this software prototype.

## License

MIT. See `LICENSE`.
