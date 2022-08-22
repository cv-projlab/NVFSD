import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from .tools_basic import filter_string, key_sort_by_numbers, tprint
from .tools_path import create_mid_dirs


########################################## VISUALIZATION ##########################################

def save_img(img: np.ndarray, img_name: str):
    cv2.imwrite(img_name, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def show_img_plt(img: np.ndarray, figsize=(10, 10), cmap="gray"):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.show()


def show_img_cv2(img: np.ndarray, wait_time=None):
    cv2.imshow("Image", img)
    if wait_time is not None:
        cv2.waitKey(wait_time)


def read_img_cv2(path:str):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def get_subplots_shape(img_no: int) -> Tuple[int, int]:
    p = np.arange(1, img_no)

    div = img_no / p
    div_int = div[div == div.astype(int)]

    closest_value = min(div_int, key=lambda list_value: abs(list_value - np.sqrt(img_no)))

    lines = closest_value
    cols = img_no / lines

    return int(lines), int(cols)


def view_multi_frames_plt(
    img_arr: np.ndarray,
    figsize: Tuple[int, int] = (15, 15),
    cmap: str = "gray",
    figshape: Tuple[int, int] = (None, None),
    subplot=None,
    transpose: str = "NHWC",
    imgs_title_list = None
) -> bool:
    """ View multiple images in the same plot based in an image array usint matplotlib

    Args:
        img_arr (np.ndarray): shape: (image_number, H, W, channels)
        figsize (Tuple[int, int], optional): _description_. Defaults to (15, 15).
        cmap (str, optional): _description_. Defaults to "gray".
        figshape (Tuple[int, int], optional): _description_. Defaults to (None, None).
        subplot (_type_, optional): _description_. Defaults to None.
        transpose (str, optional): _description_. Defaults to "NHWC".

    Returns:
        bool: _description_
    """
    assert len(transpose) == 4 and all([i in "NHWC" for i in transpose.upper()]), ""
    mapping = {"C": 0, "N": 1, "H": 2, "W": 3}  # data from dataloader shape: (channels, noframes, H, W)
    shape = tuple([mapping[letter.upper()] for letter in transpose])
    
    img_arr = np.transpose(img_arr, axes=shape)
    
    # raise NotImplementedError
    img_no = img_arr.shape[0]
    assert imgs_title_list is None or len(imgs_title_list) == img_no
    
    lines, cols = get_subplots_shape(img_no)
    
    if all(figshape) is True and figshape[0] * figshape[1] == img_no:
        lines = figshape[0]; cols = figshape[1]
        
    # print(lines, cols)    
    
    _, axs = plt.subplots(lines, cols, figsize=figsize) if subplot is None else subplot

    frame_no = 0
    for l in range(lines):
        for c in range(cols):
            axis=(l,c) if lines != 1 != cols else max(l,c)
            axs[axis].set_title(imgs_title_list[frame_no] if imgs_title_list else f'Frame {frame_no}')
            axs[axis].imshow( img_arr[frame_no, ...] , cmap=cmap) # shape: (H, W, Channels)
            axs[axis].axis('off')
            frame_no+=1
    
    plt.show()
   
    k=cv2.waitKeyEx(20)
    if k in [ord(k_) for k_ in 'qwertyuiopasdfghjklçzxcvbnm']:
        return True
    return False
            
            
def view_multi_frames_cv2(img_arr, wait_time=1, transpose='NHWC'):
    """ View frames with opencv. Only for grayscale

        Args:
            img_arr (np.ndarray): images clip; shape: (batch, H, W, channels)
        
        """
        
    assert len(transpose) == 4 and all([i in "NHWC" for i in transpose.upper()]), ""
    mapping = {"C": 0, "N": 1, "H": 2, "W": 3}  # data from dataloader shape: (channels, noframes, H, W)
    shape = tuple([mapping[letter.upper()] for letter in transpose])
    
    img_arr = np.transpose(img_arr, axes=shape)
    
    img_no = img_arr.shape[0]
    lines, cols = get_subplots_shape(img_no)
    
    himgs=[np.hstack(img_arr[c*cols:c*cols+cols,...]) for c in range(lines)]
    vimgs=np.vstack(himgs)
    cv2.imshow('Frames',vimgs)
    
    k=cv2.waitKeyEx(wait_time)
    if k in [ord(k_) for k_ in 'qwertyuiopasdfghjklçzxcvbnm']:
        return True
    return False


def view_frame_plt(img, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.show()
    
def view_frame_cv2(img, wait_time=0):
    cv2.imshow('img', img)
    cv2.waitKey(wait_time)
    

def save_frames(batch_X, lines=3, cols=4, format_='svg'):
    # frames = math.sqrt(framesPerClip)
    frame_counter = 1
    fig, ax = plt.subplots(lines, cols)
    fig = plt.figure()

    for _ in range(lines):
        for _ in range(cols):
            img = batch_X[0][0][frame_counter].cpu()
            ax = fig.add_subplot(lines, cols, frame_counter)
            ax.set_title(f'Frame {frame_counter}')
            plt.axis('off')
            plt.imshow(img)
            
            frame_counter += 1
    
    fig.savefig('frames', format=format_)
    
    

def save_img(img: np.ndarray, img_name: str):
    cv2.imwrite(img_name, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_multi_img(frames: np.ndarray, n: int, img_base_name: str, num_len=4):
    """ Save images

        Args:
            frames (np.ndarray): stacked frames
            n (int): number of frames
            img_base_name (str): images name before adding '_[images_number].png'
            
        TODO: ADD NUM LEN TO SAVE NAME
        """
    for j in range(n):
        save_img(img=(frames[:,:,j].astype('uint8')), 
                 img_name=img_base_name + f"_{j:04}.png" )
    
    
def get_frames_from_video(path: str):
    assert os.path.isfile(path), f'File {path} does not exist'
    capture = cv2.VideoCapture(path) 
    
    frames=[]   
    success, frame = capture.read()
    while success:
        frames.append(frame)
        success, frame = capture.read()
        
    return frames

def extract_frames_from_video(folder:str='s3dfm'):
    video_paths = sorted(glob(f'/home/andregraca/DATA/videos/{folder}/**/video.avi', recursive=True), key=key_sort_by_numbers)
    print(len(video_paths))
    video_paths = list(filter(filter_string, video_paths))
    print(len(video_paths))
    
    for path in tqdm(video_paths, ncols=80):
        new_path = str(Path(path).parent).replace('videos', 'images')
        create_mid_dirs(new_path)
        
        if len(os.listdir(new_path)) >= 500:
            tprint(f'Path {new_path} already has files')
            continue
        
        tprint(f'Adding files to {new_path}')
        
        images = get_frames_from_video(path)
        for i, img in enumerate(images):
            # print(img.shape)
            img_name = f'image{i:03}.png'
            cv2.imwrite(os.path.join(new_path, img_name), img[..., 0])
            


def get_pad_size(shape = 600):
    count = 0
    while True:
        pad = .5 * ((224*count)-shape)
        if pad >=0:
            break
        count += 1
    
    return int(pad)



### 

def create_acc_loss_graph(out_dir, model_log_file, dataset, BATCH_SIZE, mov_avg_N = 250):
    contents = open(model_log_file, "r").read().split('\n')

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    acc_mean = []
    loss_mean = []

    val_acc_mean = []
    val_loss_mean = []


    counterPerEpoch = np.ceil(dataset['clips'].shape[0]/BATCH_SIZE)

    counter = 0
    for c in contents:
        try:
            acc, loss, val_acc, val_loss = c.split(",")
        except:
            # print('End of file')
            pass
        else:
            times.append(float(counter/counterPerEpoch))
            counter += 1

            accuracies.append(float(acc))
            losses.append(float(loss))

            acc_mean.append(sum(accuracies[counter-mov_avg_N:counter]) / len(accuracies[counter-mov_avg_N:counter]))
            loss_mean.append(sum(losses[counter-mov_avg_N:counter]) / len(losses[counter-mov_avg_N:counter]))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

            val_acc_mean.append(sum(val_accs[counter - mov_avg_N:counter]) / len(val_accs[counter - mov_avg_N:counter]))
            val_loss_mean.append(sum(val_losses[counter - mov_avg_N:counter]) / len(val_losses[counter - mov_avg_N:counter]))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    # ax1.plot(times, accuracies, label="acc")
    # ax1.plot(times, val_accs, label="val_acc")
    ax1.plot(times, acc_mean, label="acc_mean")
    ax1.plot(times, val_acc_mean, label="val_acc_mean")
    ax1.legend()    # loc=2)

    # ax2.plot(times, losses, label="loss")
    # ax2.plot(times, val_losses, label="val_loss")
    ax2.plot(times, loss_mean, label="loss_mean")
    ax2.plot(times, val_loss_mean, label="val_loss_mean")
    ax2.legend()    # loc=2)

    plt.savefig(os.path.join(out_dir, f'{Path(model_log_file).stem}_mva{mov_avg_N}.png'))
    # plt.savefig(os.path.join(out_dir, f'logs_graph_mva{mov_avg_N}.svg'))



def create_acc_loss_graph_new(out_dir, model_log_file, dataset_length, BATCH_SIZE, mov_avg_N = 250):
    contents = open(model_log_file, "r").read().split('\n')

    
    
    
    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    acc_mean = []
    loss_mean = []

    val_acc_mean = []
    val_loss_mean = []


    counterPerEpoch = np.ceil(dataset_length/BATCH_SIZE)

    counter = 0
    for c in contents:
        try:
            acc, loss, val_acc, val_loss = c.split(",")
        except:
            # print('End of file')
            pass
        else:
            times.append(float(counter/counterPerEpoch))
            counter += 1

            accuracies.append(float(acc))
            losses.append(float(loss))

            acc_mean.append(sum(accuracies[counter-mov_avg_N:counter]) / len(accuracies[counter-mov_avg_N:counter]))
            loss_mean.append(sum(losses[counter-mov_avg_N:counter]) / len(losses[counter-mov_avg_N:counter]))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))

            val_acc_mean.append(sum(val_accs[counter - mov_avg_N:counter]) / len(val_accs[counter - mov_avg_N:counter]))
            val_loss_mean.append(sum(val_losses[counter - mov_avg_N:counter]) / len(val_losses[counter - mov_avg_N:counter]))

    fig = plt.figure()

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)


    ax1.plot(times, acc_mean, label="acc_mean")
    ax1.plot(times, val_acc_mean, label="val_acc_mean")
    ax1.legend()    
    ax2.plot(times, loss_mean, label="loss_mean")
    ax2.plot(times, val_loss_mean, label="val_loss_mean")
    ax2.legend()    
    
    plt.show()
    # plt.savefig(os.path.join(out_dir, f'{Path(model_log_file).stem}_mva{mov_avg_N}.png'))
    
    
    # plt.savefig(os.path.join(out_dir, f'logs_graph_mva{mov_avg_N}.svg'))


    
    

if __name__ == '__main__':
    # model_name='model_2022-03-03_19h13m21s_EXP13_DS4_stride1_nClasses10_Dur500_AETS_40ms_bestvalaccloss_ep17.pth'
    model_name='/home/andregraca/NebFIR/data/out/model_logs/model_EXP22_2022-04-08_11h05m31s_s1_c76_dur500_AETS_40ms.txt'
    create_acc_loss_graph_new()
