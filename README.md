# Camera View Transition
This repository contains code for my Master's thesis Camera View Transition. Example output can be found at [project's website](https://jakubhalmes.com/camera_view_transition/).

## Installation
Clone the repository and create a conda environment using the provided `conda.yaml` file.
```bash
git clone git@github.com:jac08h/camera_view_transition.git
cd camera_view_transition
conda env create -f conda.yaml
conda activate camera_view_transition
```

Download the data from [Google Drive](https://drive.google.com/file/d/15_9d1o658IrzPvk6eZvANAbgZtXQ196r) and place it in `dataset` folder.
```bash
gdown 15_9d1o658IrzPvk6eZvANAbgZtXQ196r
mkdir dataset
unzip dataset.zip -d dataset
```

## Dataset
The dataset is a collection of frames showing the same scene from different camera angles. It is based on the [SoccerNet dataset](https://www.soccer-net.org/). Each scene is represented by three files:

* Frame image: PNG image.
* Segmentation: LabelMe annotations in JSON format. It has to include `person` class for each person in the scene's foreground, usually players and referees, but not fans. Optionally, it can include `ball` class, and `graphics` class for any graphics that are not part of the scene, such as scoreboards, logos, etc.
* Camera calibration: XML files with camera calibration.

All three files are located in the same folder and have the same name, except for the extension. For example for event `3_4`, the files have to be named `3_4.png`, `3_4.json`, and `3_4.xml`.

## Usage
To generate transition, use the following command:
```
usage: generate-transition [-h] --root_dir ROOT_DIR --config CONFIG --input_event_ids INPUT_EVENT_IDS [INPUT_EVENT_IDS ...] [--duration DURATION] [--output_file OUTPUT_FILE]
                           [--padding PADDING] [--padding_source PADDING_SOURCE]

Generate a target view for a given camera based on input frames.

options:
  -h, --help            show this help message and exit
  --root_dir ROOT_DIR   Root directory of the dataset
  --config CONFIG       Path to the configuration file
  --input_event_ids INPUT_EVENT_IDS [INPUT_EVENT_IDS ...]
                        List of input event IDs
  --duration DURATION   Duration of the transition in seconds
  --output_file OUTPUT_FILE
                        Path to the output video file
  --padding PADDING     How long to display the start and end frames in seconds
  --padding_source PADDING_SOURCE
                        Source of the padding frame: with_graphics or without_graphics
```

For example:
```bash
generate-transition --root_dir "dataset/soccernet/england_epl/2015-2016/2016-04-09 - 19-30 Manchester City 2 - 1 West Brom" --config configs/config.yaml --input_event_ids 1 1_2 --duration 2 --output_file output/transition.mp4 --padding 1 --padding_source with_graphics
```

For configuration details, see `configs\config.yaml`.