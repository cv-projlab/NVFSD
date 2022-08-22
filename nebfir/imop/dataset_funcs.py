import os
import string
from enum import Enum, auto
from glob import glob
from pathlib import Path
import time
from typing import List
from scipy.io import loadmat

import cv2
import numpy as np
from tqdm import tqdm

from .or_scaling import OR_image_scale
from ..tools.tools_basic import multi_split, tprint, key_sort_by_numbers, str2int_list
from ..tools.tools_visualization import save_multi_img, get_pad_size
from ..tools.tools_path import create_mid_dirs, join_paths

from .events import (BaseEventsRepresentation, create_ts_csv, sort_events)


class FrameType(Enum):
    events = 'EventsRepresentation'
    grayscale = 'Images'


def create_clips(folder: str, frame_type: FrameType, frame_rep_type:str='AETS_40ms', view_tqdm=True, base_path="data/datasets", DRY_RUN=False):
    """ Create image clips in format ( channels, frame_no, 224, 224 ) to use as the model input

    Args:
        folder (str): Folder to create events representation from. i.e: s3dfm, deepfakes_v1
        frame_type (FrameType): events or grayscale
        frame_rep_type (str, optional): frames representation type. Original grayscale frames -> grayscale; Event frames -> [AETS, SAE, SNN, TBR,...]_[40, 20, ...]ms. Defaults to 'AETS_40ms'.
        view_tqdm (bool, optional): View progress bar. Defaults to True.
        base_path (str, optional): ROOT Path to dataset folder. ex: 'base_path/SynFED'. Defaults to "data/inp/dataset".

    """

    #### PATHS ####
    frames_dir = str(Path(base_path) / f"SynFED/{frame_type.value}/{folder}/{frame_rep_type if frame_type is FrameType.events else ''}")
    clips_dir  = str(Path(base_path) / f"SynFED/Clips/{folder}/{frame_rep_type}")

    assert (frame_type is FrameType.events and frame_rep_type != 'original') or (frame_type is FrameType.grayscale and frame_rep_type == 'original'), f'Wrong representation type: Got {frame_type.name} with rep {frame_rep_type}'
    assert os.path.isdir(frames_dir), f'{frames_dir} does not exist!'
    
    create_mid_dirs(clips_dir) if not DRY_RUN else print(f'Creating directory: {clips_dir}')

    print("frames_dir:", frames_dir, "\nclips_dir:", clips_dir)
    
    dir_paths = sorted(glob(join_paths(frames_dir, "u*", "r*")), key=key_sort_by_numbers)  # Get all users paths

    
    if view_tqdm:
        pbar = tqdm(dir_paths, desc="User: --; Recording: --")

    for dir_ in dir_paths:
        user, rec = dir_.split("/")[~1:]
        save_clips_dir = join_paths(clips_dir, f"clip_{user}{rec}")

        if view_tqdm:
            pbar.set_description_str(f"User: {user: >3} - Recording: {rec: >3}")
            pbar.update()
            
        if os.path.isfile(join_paths(clips_dir, f"clip_{user}{rec}.npy")):
            tprint(join_paths(clips_dir, f"clip_{user}{rec}.npy")+' already exists')
            continue

        paths = sorted(glob(join_paths(frames_dir, user, rec, "*")))  # Get all images from user path
        frames = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in sorted(paths)]  # Read all images


        if DRY_RUN:
            tprint(f'Resizing {len(frames)} frames')
            tprint(f'Saving clip to {save_clips_dir}')
            continue

        clip_list = []
        for frame in frames:
            # Rescale image to 224,224
            if frame_type is FrameType.events:
                pad_size = get_pad_size(frame.shape[0])
                padded_frame = np.pad(array=frame, pad_width=((pad_size, pad_size), (pad_size, pad_size)), constant_values=0,)
                resized_frame = OR_image_scale(padded_frame, 224 / padded_frame.shape[0])
            elif frame_type is FrameType.grayscale:
                resized_frame = cv2.resize(frame, (224, 224))

            # Stack frames to form a clip
            min_arg = np.argmin(resized_frame.shape)
            new_shape = (
                min_arg,
                *np.delete(np.arange(resized_frame.ndim), min_arg),
            )  # (dim_min_arg, dim_1, dim_2, ...)
            clip = (
                np.transpose(resized_frame, new_shape) if resized_frame.ndim > 2 else np.expand_dims(resized_frame, axis=0)
            )  # if img_dim>2 transpose to new shape, else expand dimension to be in shape ( channels, 224, 224 )
            clip_list.append(clip)  # shape: [( channels, 224, 224 ), ... ]

        cliparray = np.stack(clip_list, 1)  #  shape: ( channels, frame_no, 224, 224 )
        # print(cliparray.shape)

        np.save(save_clips_dir, cliparray)

    if view_tqdm:
        pbar.clear()


def create_event_frames(folder: str, events_representation: BaseEventsRepresentation, dT:int=40_000,  view_tqdm=True, base_path='data/datasets', DRY_RUN=False):
    """ Create event representation frames from given event files 

    Args:
        folder (str): Folder to create events representation from. i.e: s3dfm, deepfakes_v1
        events_representation (BaseEventsRepresentation): Events representation. i.e: AETS, SNN, SAE, TBR, ...
        dT (int, optional): Integration time in micro-seconds (μs). Defaults to 40_000.
        view_tqdm (bool, optional): View progress bar. Defaults to True.
        base_path (str, optional): ROOT Path to dataset folder. ex: 'base_path/SynFED'. Defaults to '/mnt/DATADISK/Datasets/face/'.
        DRY_RUN (bool, optional): Dry run. Defaults to False.
    """
    events_rep = events_representation(frame_shape=(600, 600), dT=dT)  # frame_shape=(600,600),  dT = 120000
    # events_rep.toggle_debug_mode()
    #### PATHS ####
    events_dir = str(Path(base_path) / f"SynFED/Events/{folder}")
    frames_dir = str(Path(base_path) / f"SynFED/EventsRepresentation/{folder}")
    
    print("events_dir:", events_dir, "\nframes_dir:", frames_dir)
    
    # make new dir
    reppath = f"{events_rep.__class__.__name__}_{events_rep.dT/1000:02.0f}ms"
    save_frames_dir = join_paths(frames_dir, reppath)
    create_mid_dirs(save_frames_dir) if not DRY_RUN else print(f'Creating directory: {save_frames_dir}')

    events_txt_paths = sorted(glob(f"{events_dir}/**/events.txt", recursive=True), key=key_sort_by_numbers)
    events_csv_paths = sorted(glob(f"{events_dir}/**/pc.csv", recursive=True), key=key_sort_by_numbers)
    ## Create events .csv from .txt if sizes mismatch
    if len(events_txt_paths) != len(events_csv_paths):
        tprint('Creating events csv files ')
        if not DRY_RUN:
            [create_ts_csv(path_to_events=path) for path in tqdm(events_txt_paths, desc="Creating pc.csv from events.txt")]  # Create pc.csv files from events.txt
        events_csv_paths = sorted(glob(f"{events_dir}/**/pc.csv", recursive=True), key=key_sort_by_numbers)

    if view_tqdm:
        frames_bar = tqdm(
            total=10, position=1, desc=f"PNG frame: ----", colour="green", bar_format="{desc} {percentage:3.0f}%|{bar:24}| [{elapsed}<{remaining}]",
        )

        total_bar = tqdm(events_csv_paths, position=0, desc="User: --; Recording: --")

    tprint('Creating events frames ')
    for events_csv_path in events_csv_paths:
        user, rec = Path(events_csv_path).parts[~2:~0]
        basename = f"frame_{user}{rec}"
        save_path = join_paths(save_frames_dir, user, rec)

        # tprint(join_paths(save_frames_dir, user, rec, f"{basename}_{0:04}.png"))
        if view_tqdm:
            total_bar.set_description_str(f"User: {user: <3}; Recording: {rec: <3}")
            total_bar.update()

        # If frames already exist continue
        if os.path.isfile(join_paths(save_frames_dir, user, rec, f"{basename}_{0:04}.png")):
            continue

        # If the events_csv_path exists 
        if os.path.isfile(events_csv_path) and not DRY_RUN:

            if view_tqdm:
                frames_bar.reset()
            new_arr, _ = sort_events(events_csv_path)

            # Convert events to event representation frames
            events_rep.convert_events_to_frames(arr=new_arr, bar=frames_bar if view_tqdm else None)

            # SAVE FRAMES
            create_mid_dirs(save_path)
            save_multi_img(frames=events_rep.FRAMES, n=events_rep.FRAME_NO , img_base_name=os.path.join(save_path, basename))

        elif DRY_RUN:
            tprint(f'Creating directories: {save_path}; Saving frames to the same directory')
            # time.sleep(.05)

        elif not os.path.isfile(events_csv_path):
            tprint(f'{events_csv_path} does not exist!')


    if view_tqdm:
        frames_bar.clear()
        total_bar.clear()



def create_event_frames_nvfsd(events_representation: BaseEventsRepresentation, dT:int=40_000,  view_tqdm=True, base_path='data/datasets', DRY_RUN=False):
    """ Create event representation frames from given event files 

    Args:
        folder (str): Folder to create events representation from. i.e: s3dfm, deepfakes_v1
        events_representation (BaseEventsRepresentation): Events representation. i.e: AETS, SNN, SAE, TBR, ...
        dT (int, optional): Integration time in micro-seconds (μs). Defaults to 40_000.
        view_tqdm (bool, optional): View progress bar. Defaults to True.
        base_path (str, optional): ROOT Path to dataset folder. ex: 'base_path/SynFED'. Defaults to 'data/inp/dataset'.
        DRY_RUN (bool, optional): Dry run. Defaults to False.
    """
    events_rep = events_representation(frame_shape=(800, 1280), dT=dT)  # frame_shape=(600,600),  dT = 120000
    # events_rep.toggle_debug_mode()
    
    mat_contents = loadmat(os.path.join('data/inp/', "xlocation_40users.mat"))
    xlocation = mat_contents.get('xlocation')
    
    #### PATHS ####
    events_dir = str(Path(base_path) / f"NVFSD/Events/")
    frames_dir = str(Path(base_path) / f"NVFSD/EventsRepresentation/")
    
    print("events_dir:", events_dir, "\nframes_dir:", frames_dir)
    
    # make new dir
    reppath = f"{events_rep.__class__.__name__}_{events_rep.dT/1000:02.0f}ms"
    save_frames_dir = join_paths(frames_dir, reppath)
    create_mid_dirs(save_frames_dir) if not DRY_RUN else print(f'Creating directory: {save_frames_dir}')

    events_csv_paths = sorted(glob(f"{events_dir}/**/pc.csv", recursive=True))
   
    if view_tqdm:
        frames_bar = tqdm(
            total=10, position=1, desc=f"PNG frame: ----", colour="green", bar_format="{desc} {percentage:3.0f}%|{bar:24}| [{elapsed}<{remaining}]",
        )

        total_bar = tqdm(events_csv_paths, position=0, desc="User: --; Recording: --")

    tprint('Creating events frames ')
    for events_csv_path in events_csv_paths:
        user, task, rec = Path(events_csv_path).parts[~3:~0]
        seps= string.ascii_letters
        remove_letters = lambda x: multi_split(x, seps, map_func=str2int_list)[0]

        basename = f"frame_{user}_{task}_{rec}"
        
        
        save_path = join_paths(save_frames_dir, user, task, rec)

        # tprint(join_paths(save_frames_dir, user, rec, f"{basename}_{0:04}.png"))
        if view_tqdm:
            total_bar.set_description_str(f"User: {remove_letters(user): <3}; Task: {remove_letters(task): <3}; Recording: {remove_letters(rec): <3}")
            total_bar.update()

        # If frames already exist continue
        if os.path.isfile(join_paths(save_frames_dir, user, task, rec, f"{basename}_{0:04}.png")):
            continue

        # If the events_csv_path exists 
        if os.path.isfile(events_csv_path) and not DRY_RUN:

            if view_tqdm:
                frames_bar.reset()
                
            new_arr, _ = sort_events(events_csv_path)

            # Convert events to event representation frames
            events_rep.convert_events_to_frames(arr=new_arr, bar=frames_bar if view_tqdm else None)

              
            CROPPED_FRAMES = events_rep.get_cropped_frames(x=200 if remove_letters(task)>3 else int(xlocation[remove_letters(user), remove_letters(task)-1, remove_letters(rec)-1]))
            # SAVE FRAMES
            create_mid_dirs(save_path)
            save_multi_img(frames=CROPPED_FRAMES, n=events_rep.FRAME_NO , img_base_name=os.path.join(save_path, basename))

        elif DRY_RUN:
            tprint(f'Creating directories: {save_path}; Saving frames to the same directory')
            # time.sleep(.05)

        elif not os.path.isfile(events_csv_path):
            tprint(f'{events_csv_path} does not exist!')


    if view_tqdm:
        frames_bar.clear()
        total_bar.clear()


def create_clips_nvfsd(frame_type: FrameType, frame_rep_type:str='AETS_40ms', view_tqdm=True, base_path="data/datasets", DRY_RUN=False):
    """ Create image clips in format ( channels, frame_no, 224, 224 ) to use as the model input

    Args:
        folder (str): Folder to create events representation from. i.e: s3dfm, deepfakes_v1
        frame_type (FrameType): events or grayscale
        frame_rep_type (str, optional): frames representation type. Original grayscale frames -> grayscale; Event frames -> [AETS, SAE, SNN, TBR,...]_[40, 20, ...]ms. Defaults to 'AETS_40ms'.
        view_tqdm (bool, optional): View progress bar. Defaults to True.
        base_path (str, optional): ROOT Path to dataset folder. ex: 'base_path/SynFED'. Defaults to "data/inp/dataset".

    """

    #### PATHS ####
    frames_dir = str(Path(base_path) / f"NVFSD/EventsRepresentation/{frame_rep_type}")
    clips_dir  = str(Path(base_path) / f"NVFSD/Clips/{frame_rep_type}")

    assert os.path.isdir(frames_dir), f'{frames_dir} does not exist!'
    
    create_mid_dirs(clips_dir) if not DRY_RUN else print(f'Creating directory: {clips_dir}')

    print("frames_dir:", frames_dir, "\nclips_dir:", clips_dir)
    
    dir_paths = sorted(glob(os.path.join(frames_dir, "u*", 't*', "r*"), recursive=True))  # Get all users paths
    
    if view_tqdm:
        pbar = tqdm(dir_paths, desc="User: --; Task: ---; Recording: --")

        clips_bar = tqdm(
            total=10, position=1, desc=f"PNG frame: ----", colour="green", bar_format="{desc} {percentage:3.0f}%|{bar:24}| [{elapsed}<{remaining}]",
        )


    for dir_ in dir_paths:
        user, task, rec = dir_.split("/")[~2:]
        save_clips_dir = join_paths(clips_dir, f"clip_{user}{task}{rec}")

        if view_tqdm:
            pbar.set_description_str(f"User: {user: >3} - Task: {task: >3} - Recording: {rec: >3}")
            pbar.update()
            
        if os.path.isfile(join_paths(clips_dir, f"clip_{user}{task}{rec}.npy")):
            tprint(join_paths(clips_dir, f"clip_{user}{task}{rec}.npy")+' already exists')
            continue

        paths = sorted(glob(join_paths(frames_dir, user, task, rec, "*")))  # Get all images from user path
        frames = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in sorted(paths)]  # Read all images


        if DRY_RUN:
            tprint(f'Resizing {len(frames)} frames')
            tprint(f'Saving clip to {save_clips_dir}')
            continue

        
        if view_tqdm:
            clips_bar.reset()
            clips_bar.total = len(frames)
                

        clip_list = []
        for i, frame in enumerate(frames):
            # Rescale image to 224,224
            if frame_type is FrameType.events:
                pad_size = get_pad_size(frame.shape[0])
                padded_frame = np.pad(array=frame, pad_width=((pad_size, pad_size), (pad_size, pad_size)), constant_values=0,)
                resized_frame = OR_image_scale(padded_frame, 224 / padded_frame.shape[0])
            elif frame_type is FrameType.grayscale:
                resized_frame = cv2.resize(frame, (224, 224))

            # Stack frames to form a clip
            min_arg = np.argmin(resized_frame.shape)
            new_shape = (
                min_arg,
                *np.delete(np.arange(resized_frame.ndim), min_arg),
            )  # (dim_min_arg, dim_1, dim_2, ...)
            clip = (
                np.transpose(resized_frame, new_shape) if resized_frame.ndim > 2 else np.expand_dims(resized_frame, axis=0)
            )  # if img_dim>2 transpose to new shape, else expand dimension to be in shape ( channels, 224, 224 )
            clip_list.append(clip)  # shape: [( channels, 224, 224 ), ... ]

            if clips_bar is not None:
                clips_bar.set_description_str(f"Clip : {int(i): >4}")
                clips_bar.update()

        cliparray = np.stack(clip_list, 1)  #  shape: ( channels, frame_no, 224, 224 )
        # print(cliparray.shape)

        np.save(save_clips_dir, cliparray)

    if view_tqdm:
        pbar.clear()
        clips_bar.clear()



if __name__ == "__main__":
    # folder = "deepfakes_v1"  # deepfakes_flow s3dfm_flow s3dfm deepfakes_v3

    # create_clips(folder=folder, frame_type=FrameType.grayscale)

    # create_event_frames(folder=folder, events_representation=AETS)
    # create_clips(folder=folder, frame_type=FrameType.events)



    # frame_type = FrameType.events
    # print(frame_type is FrameType.events)
    # print(frame_type is FrameType.grayscale)




    # arr = np.load("DATA/DatasetsClips/deepfakes_v1/grayscale/clip_u0r0.npy")[:, :25, ...]
    # # arr = np.load("DATA/DatasetsClips/s3dfm/AETS_40ms/clip_u0r0.npy")
    # print(arr.shape)
    # view_multi_frames_plt(arr)

    pass
