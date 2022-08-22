# NebFIR EXTRA

Allways assuming user is in base directory ```NebFIR/```

## Table of contents

- [NebFIR EXTRA](#nebfir-extra)
  - [Table of contents](#table-of-contents)
  - [1. CREATE EVENT FRAMES AND CLIPS](#1-create-event-frames-and-clips)
      - [Vizualize steps to create only event frames for SynFED](#vizualize-steps-to-create-only-event-frames-for-synfed)
      - [Create only clips for NVFSD without console output](#create-only-clips-for-nvfsd-without-console-output)
      - [Create both event frames AND clips AND choose the base dataset path AND choose the events representation for SynFED](#create-both-event-frames-and-clips-and-choose-the-base-dataset-path-and-choose-the-events-representation-for-synfed)
    - [USAGE: create_event_frames_and_clips](#usage-create_event_frames_and_clips)
  - [2. CREATE LISTS](#2-create-lists)
      - [Create a list with all available data for SynFED](#create-a-list-with-all-available-data-for-synfed)
      - [Create train and test lists for SynFED from existing list with all available data](#create-train-and-test-lists-for-synfed-from-existing-list-with-all-available-data)
      - [Create a list with all available data and train and test lists for SynFED](#create-a-list-with-all-available-data-and-train-and-test-lists-for-synfed)
      - [Create lists with different list options for SynFED](#create-lists-with-different-list-options-for-synfed)
      - [Create lists with different list options for NVFSD](#create-lists-with-different-list-options-for-nvfsd)
      - [Create lists without real dynamics for SynFED](#create-lists-without-real-dynamics-for-synfed)
      - [Create list and generate config file](#create-list-and-generate-config-file)
    - [USAGE: create_lists](#usage-create_lists)
  - [3. TRAIN](#3-train)
      - [Train model with different batchize, epochs and device](#train-model-with-different-batchize-epochs-and-device)
      - [Test model with different list](#test-model-with-different-list)
    - [USAGE: runner](#usage-runner)

&nbsp;

## 1. CREATE EVENT FRAMES AND CLIPS

#### Vizualize steps to create only event frames for SynFED

  ```bash
  python create_event_frames_and_clips.py SynFED --frames --DRY_RUN
  ```

#### Create only clips for NVFSD without console output

  ```bash
    python create_event_frames_and_clips.py NVFSD --clips --SILENT
  ```
  
#### Create both event frames AND clips AND choose the base dataset path AND choose the events representation for SynFED

  ```bash
    python create_event_frames_and_clips.py SynFED --frames --clips -p /path/to/SynFED -r FRQ
  ```
  
&nbsp;

### USAGE: create_event_frames_and_clips

```bash
usage: create_event_frames_and_clips.py [-h] [--frames] [--clips] [-s] [-d]
                                        [--summary] [-p PATH]
                                        [-r {AETS,FRQ,SAE,SNN,TBR}]
                                        [-t {events,grayscale}]
                                        {SynFED,NVFSD}

positional arguments:
  {SynFED,NVFSD}        Dataset

optional arguments:
  -h, --help            show this help message and exit
  --frames, --create_frames, --create_event_frames
                        Create event frames from event files
  --clips, --create_clips
                        Create frame clips from event frames
  -s, -S, --silent, --SILENT
                        Silent data creation
  -d, -D, --dry_run, --DRY_RUN, --dryrun, --DRYRUN
                        Dry run
  --summary             Prints options summary
  -p PATH, --path PATH, --PATH PATH, --root_path PATH
                        Data root path to events folder; i.e: "<PATH>/SynFED"
  -r {AETS,FRQ,SAE,SNN,TBR}, --events_representation {AETS,FRQ,SAE,SNN,TBR}
                        Events representation
  -t {events,grayscale}, --frame_type {events,grayscale}
                        Frame type
```
&nbsp;

## 2. CREATE LISTS

#### Create a list with all available data for SynFED

  ```bash
  python create_lists.py SynFED create_base_list --summary
  ```

#### Create train and test lists for SynFED from existing list with all available data

  ```bash
  python create_lists.py SynFED create_list --summary
  ```

#### Create a list with all available data and train and test lists for SynFED

  ```bash
  python create_lists.py SynFED create_base_list create_list --summary
  ```

#### Create lists with different list options for SynFED

  ```bash
  python create_lists.py SynFED create_list --frame_type events_aets40 --userno 30 --recno 10 --fakeno 15 --train_split_frac .8 --summary
  ```

#### Create lists with different list options for NVFSD

  ```bash
  python create_lists.py NVFSD create_list --userno 40 --combination A1B1 --train_split_frac .7 --summary
  ```

#### Create lists without real dynamics for SynFED

  ```bash
  python create_lists.py create_list --no_real --summary
  ```

#### Create list and generate config file

  ```bash
  python create_lists.py create_list --gen_cfg --summary
  ```
&nbsp;

### USAGE: create_lists

```bash
usage: create_lists.py [-h] [-t {events_aets40,grayscale}] [-p PATH]
                       [--userno USERNO] [--train_split_frac TRAIN_SPLIT_FRAC]
                       [-c COMBINATION] [--recno RECNO] [--fakeno FAKENO]
                       [--no-real] [--no-fake] [--summary] [--gen_conf]
                       {SynFED,NVFSD} {create_base_list,create_list}
                       [{create_base_list,create_list} ...]

positional arguments:
  {SynFED,NVFSD}        Dataset
  {create_base_list,create_list}
                        Function to run

optional arguments:
  -h, --help            show this help message and exit
  -t {events_aets40,grayscale}, --frame_type {events_aets40,grayscale}, --type {events_aets40,grayscale}
                        Frame type
  -p PATH, --path PATH, --PATH PATH, --root_path PATH
                        Data root path to dataset folder; i.e: "<PATH>/SynFED"
  --userno USERNO       Number of users
  --train_split_frac TRAIN_SPLIT_FRAC
                        Train split fraction
  -c COMBINATION, --comb COMBINATION, --combination COMBINATION
                        NVFSD: Task Recording Combination
  --recno RECNO         SynFED: Real recordings number
  --fakeno FAKENO       SynFED: Fake users number
  --no-real, --no-authentic
                        SynFED: Creates train and test list without real
                        dynamics
  --no-fake, --no-impostor
                        SynFED: Creates train and test list without fake
                        dynamics
  --summary             Prints options summary
  --gen_conf, --gen_config, --generate_configuration_file
                        Automaticaly generate config file
```


&nbsp;
<div align="center">
<!-- <style> td { font-size: 10px } </style> -->

| | folder | path | frameno_list | user | rec | real |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | deepfakes_v1 | data/datasets/SynFED/Clips/deepfakes_v1/AETS_40ms/clip_u0r0.npy | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]     |  0 | 0 | 1 |
| 1 | deepfakes_v1 | data/datasets/SynFED/Clips/deepfakes_v1/AETS_40ms/clip_u0r0.npy | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]    |  0 | 0 | 1 |
| 2 | deepfakes_v1 | data/datasets/SynFED/Clips/deepfakes_v1/AETS_40ms/clip_u0r0.npy | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]   |  0 | 0 | 1 |
| 3 | deepfakes_v1 | data/datasets/SynFED/Clips/deepfakes_v1/AETS_40ms/clip_u0r0.npy | [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  |  0 | 0 | 1 |
| 4 | deepfakes_v1 | data/datasets/SynFED/Clips/deepfakes_v1/AETS_40ms/clip_u0r0.npy | [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] |  0 | 0 | 1 |
| ... | ... | ... | ... | ... | ... | ... |

Example of SynFED event list

&nbsp;
| | path | frameno_list | user | task | rec | real |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|  0 | data/datasets/NVFSD/Clips/AETS_40ms/clip_user000task01recording01.npy | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] | 0 | 1 | 1 | 1 |
|  1 | data/datasets/NVFSD/Clips/AETS_40ms/clip_user000task01recording01.npy | [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] | 0 | 1 | 1 | 1 |
| 2 | data/datasets/NVFSD/Clips/AETS_40ms/clip_user000task01recording01.npy | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] | 0 | 1 | 1 | 1 |
| 3 | data/datasets/NVFSD/Clips/AETS_40ms/clip_user000task01recording01.npy | [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] | 0 | 1 | 1 | 1 |
| 4 | data/datasets/NVFSD/Clips/AETS_40ms/clip_user000task01recording01.npy | [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] | 0 | 1 | 1 | 1 |
| ... | ... | ... | ... | ... | ... | ... |

Example of NVFSD event list

&nbsp;
<div align="left">


## 3. TRAIN

#### Train model with different batchize, epochs and device

```bash
python runner.py --cfg configs/config-0.yml -t --batch_size 16 --device cuda:1 --epochs 100
```

#### Test model with different list

```bash
python runner.py --cfg configs/config-0.yml -i --weights path/to/model/weights --il path/to/test/list
```

&nbsp;

### USAGE: runner

```bash
usage: runner.py [-h] [-c {cfg_file.yml}] [-w MODEL_WEIGHTS] [--tl {list.csv}]
                 [--il {list.csv}] [-d] [-t] [-i] [--description DESCRIPTION]
                 [--device DEVICE] [-ch {1,3}] [-b BATCH_SIZE] [-e EPOCHS]
                 [--notif]

Trainer arguments

optional arguments:
  -h, --help            show this help message and exit
  -c {cfg_file.yml}, --cfg {cfg_file.yml}, --conf {cfg_file.yml}, --config {cfg_file.yml}, --configuration {cfg_file.yml}
                        Configuration file
  -w MODEL_WEIGHTS, -p MODEL_WEIGHTS, --weights MODEL_WEIGHTS, --path MODEL_WEIGHTS
                        Model path
  --tl {list.csv}, --train_list {list.csv}
                        Choose a train list
  --il {list.csv}, --inference_list {list.csv}, --test_list {list.csv}
                        Choose a test list
  -d, --dry             Dry run
  -t, --train           Train
  -i, --inference, --test
                        Test
  --description DESCRIPTION
                        Optional brief description about the training
  --device DEVICE       Device to run the model
  -ch {1,3}, --channels {1,3}
                        Number of channels
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch Size
  -e EPOCHS, --epochs EPOCHS
                        Train epochs
  --notif, --email_error, --error_notification, --ERROR_NOTIFICATION
                        Email error notification
```

<style scoped > table { font-size: 13px; } </style>
