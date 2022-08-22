# S3DFM

PAPER:

https://www.pure.ed.ac.uk/ws/files/79659390/3D_Visual_Passcode.pdf

DATASET LINK:

https://groups.inf.ed.ac.uk/trimbot2020/DYNAMICFACES/

citation:

Jie Zhang, Robert B. Fisher; 3D visual passcode: Speech-driven 3D facial dynamics for behaviometrics, Signal Processing, 2019, 160: 164-177.

# First Order Motion Model for Image Animation (fommia)

PAPER:

https://arxiv.org/pdf/2003.00196.pdf

GITHUB:

https://github.com/AliaksandrSiarohin/first-order-model

citation:

@InProceedings{Siarohin_2019_NeurIPS,
  author={Siarohin, Aliaksandr and Lathuilière, Stéphane and Tulyakov, Sergey and Ricci, Elisa and Sebe, Nicu},
  title={First Order Motion Model for Image Animation},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  month = {December},
  year = {2019}
}

# Video to Events (v2e)

PAPER:

https://arxiv.org/pdf/2006.07722.pdf

GITHUB:

https://github.com/SensorsINI/v2e

citation:

Y. Hu, S-C. Liu, and T. Delbruck. v2e: From Video Frames to Realistic DVS Events. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), URL: https://arxiv.org/abs/2006.07722, 2021

## Process

### Dataset

The S3DFM is a dynamic 2D/3D speaking face dataset with synchronized audio.

There are 77 participants looking at the camera and each one repeats the same passphrase (Ni hao) 10 times (with a total of 770 sets of data). There are an additional 26 (*10 repetitions) participants that were moving their heads while speaking the same passphrase.

This dataset contains syncronized:

- 77 x 10 x 1 second of 500 frame per second IR intensity video (77 x 10 x 500 intensity frames, 600 x 600 pixels)
- 77 x 10 x 1 second of 500 frame per second registered depth images (77 x 10 x 500 3D point clouds, 600 x 600 pixels)
- 77 x 10 x 1 second of audio sequence with a sampling frequency of 44.1 Khz

Each file contains 10 sequences (3D & intensity & audio) from a participant. You could download them by clicking on a file and then unzipping them individually.

Each available zip file (~2.23 GB) consists of 500 Matlab *.mat files (~4.5 MB) and one *.mp3 file. Each matlab *.mat file (e.g. seq1_050.mat) contains 2 arrays of single precision type: Img(600,600) and XYZ(600,600,3). Img is the infrared intensity image at the corresponding frame. XYZ is the (x,y,z) point computed for the corresponding pixel. The mp3 file consists of approximately 44100 audio samples. The first audio sample has been synchronized previously to align with the first video frame.

### Event frames generation

We start by downloading each zip file and extract the sequence of intensity frames from the *.mat files. We generate events from these frames using the pretrained v2e network wich accepts video as input. This network computes frame intensity variation and outputs a file with timestamped events in the form: timestamp (seconds), x, y, polarity(on=1, off=0). With this files we can finnaly create event representation frames based on the AETS algorithm for each participant.

To extend the variety of data we generate deepfake videos of the S3DFM dataset participants. We use the pretrained fommia network wich takes as input an image and a video. This network applies the dynamic of the video to the source image. Based on this we create three deepfake variations:

- version 1: for each participant (77) source image we apply the dynamic of every participant (77) video, including themselves, resulting in 77 x 77 sequences of video
- version 2: for each participant (77) source image we apply the dynamic of every video sequence (10) of themselves, resulting in 77 x 10 sequences of video
- version 3: with an unique (1) source image we apply the dynamic of every video sequence (10) of every participant (77), resulting in 77 x 10 sequences of video

With the deepfake videos created we repeat the same process as for the original S3DFM dataset, obtaining event representation frames for the three deepfake versions.



* v2e options:
  -i inp_video
  -o out_events
  --unique_output_folder true --skip_video_output --no_preview
  --batch_size 4
  --davis_output --dvs_h5 None --dvs_aedat2 None --dvs_text events.txt
  --slomo_model input/SuperSloMo39.ckpt
  --dvs_exposure duration .040
  --input_frame_rate 500
  --input_slowmotion_factor 1
  --auto_timestamp_resolution true
  --pos_thres 0.3 --neg_thres 0.3 --sigma_thres 0.02
  --cutoff_hz 15 --leak_rate_hz 0 --shot_noise_rate_hz 0

* fommia options:
  --driving_video inp_video
  --source_image inp_img
  --result_video out_video
  --config config/vox-adv-256.yaml
  --checkpoint data/chkpts/vox-cpk.pth.tar
  --relative
  --adapt_scale
