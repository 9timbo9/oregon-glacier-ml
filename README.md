Setup

Install required packages:
pip install -r requirements.txt




Configuration

Before running the scripts, open config.py and set the year you want to process.
The year is controlled using the variable:

CURRENT_YEAR_IDX

Possible values:
0 = 1980 (Landsat 5)
1 = 2000 (Landsat 7)
2 = 2020 (Landsat 8)

Example:
CURRENT_YEAR_IDX = 1

This will process year 2000 data using Landsat 7 imagery.
The selected year automatically sets:

CURRENT_YEAR
LANDSAT
patches/<YEAR>/
outputs/<YEAR>/
so you do not need to manually change the paths.




Workflow
1. Create patches (optional)
python patcher.py

This script allows you to interactively select regions of interest and saves patch data for training and inference.

2. Train the model (optional)
python train_glacier_unet_pseudolabel.py

Training is not required, since a pretrained model is already provided:

models/glacier_unet_pseudolabel.pt

Training may take a long time depending on hardware.

3. Run glacier prediction
python infer_and_measure.py

This script:

loads the trained model

predicts glacier masks

calculates glacier area, length, and width

Results are saved to:

outputs/<YEAR>/glacier_measurements.csv
4. Generate glacier outline images
python overlay_glacier_outline.py

This creates visualizations showing predicted glacier boundaries over satellite imagery.

Images are saved to:

outputs/<YEAR>/outline_images/