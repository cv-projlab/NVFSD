import os
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..tools.tools_basic import find_term_in_string
from ..tools.tools_visualization import read_img_cv2

# import transformations as T
# from tools.tools_visualization import view_multi_frames_cv2, view_multi_frames_plt
    
class FileMode(Enum):
    numpy_clip = auto()
    png_img = auto()


def map_users(user_list: List[int]) -> Dict[int, int]:
    """ Maps a user list to a range of length len(user_list)

    Args:
        user_list (List): user list

    Returns:
        Dict[int, int]: user mapping; User -> Integer
    """
    values = tuple(range(len(user_list)))
    return dict(zip(sorted(user_list), values))


class EventsDataset(Dataset):
    """Events dataset"""
    
    def __init__(self, csv_file, transform=None, DEBUG=False, file_mode:FileMode = FileMode.numpy_clip):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.DEBUG = DEBUG
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.transform = transform

        self.users_list = list(self.df["user"].unique())
        self.userno = len(self.users_list)  # self.USERNO = 10

        self.user_mapping = map_users(self.users_list)

        self.file_mode = file_mode



    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.df.loc[idx]  # pd.Dataframe line -> path, frameno_list, user, rec
                                 #                      DATA/event_frames/{line.user}/{line.rec}/frame_{line.user}{line.rec}_{frameno:0>4}, frameno_list, user, rec
        
        idxs = np.array(line.frameno_list.strip("[ ]").split(","), dtype=int)  # converts string "[ 1, 2, 3, 4, 5 ]" to np.array([1,2,3,4,5])
        
        # Get subclip
        if self.file_mode is FileMode.numpy_clip:
            clip = np.load(line.path)  # shape: (no_channels, frames, H, W)
            subclip = clip[:, idxs, ...]
        
        elif self.file_mode is FileMode.png_img:
            subclip = np.array([read_img_cv2(os.path.join(line.path, f'frame_u{line.user}r{line.rec}_{frameno:0>4}.png' if not 'images' in line.path else f'image{frameno:0>3}.png')) for frameno in idxs])

            if subclip.ndim <= 3: # shape: (Time, H, W)
                subclip = np.expand_dims(subclip, 0) # shape: (Channel, Time, H, W)
        else: raise ValueError(f'File mode {self.file_mode} unexpected ! File mode must be of type [ {" | ".join([f.name for f in FileMode])} ]')


        label = self.user_mapping[line.rec] if not line.real else self.user_mapping[line.user] 
        
        folder = find_term_in_string(string=line.path)
        # label = self.user_mapping[line.rec] if folder == 'deepfakes_v1' else self.user_mapping[line.user] 
            
        
        if self.transform :
            subclip = self.transform(subclip)
            
            
        sample = {
            "clip": subclip,
            "label": label,
            "ohlabel": np.eye(self.userno)[label],
            "user": line.user,
            'task':line.get('task', 'THIS LIST DOES NOT CONTAIN TASK COLUMNS'), # SynFED does not have tasks
            "recording": line.rec,
            'folder': folder, # NVFSD does not have folder
            'real':line.real,
        }
        
        if self.DEBUG:
            # print(line)
            # print(line.path.split('/')[2])
            # print('clip', clip.shape)
            # print('idxs', idxs)
            # print('subclip', subclip.shape)
            # print(label)
            # print(folder)
            # print('idx', idx)
            # print('label',label, '| user',line.user, '| rec', line.rec, '| folder', folder, )
            print('label',sample['label'], '| user',sample['user'], '| rec', sample['recording'], '| folder', sample['folder'], )
            # print(sample)
            pass
        


        return sample

    def get_class_data(self):
        raise NotImplementedError
        data = {
            "csv_file": self.csv_file,
            "user_number": self.userno,
            "user_mapping": self.user_mapping,
            "dataset_length": len(self),
        }
        return data

    def __repr__(self) -> str:
        return f"Events Dataset\n  csv file: {self.csv_file}\n  user number: {self.userno}\n  user mapping: {self.user_mapping}\n  dataset length: {len(self)}"


class EventsDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=False, drop_last=False, **kwargs) -> None:
        super().__init__( dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, **kwargs)        

    def __repr__(self) -> str:
        return super().__repr__() + f'\nSize: {len(self)}'






class EventsDatasetV2(Dataset):
    """Events dataset"""
    
    def __init__(self, csv_file:str='data/inp/lists/SynFED_df_all_events_aets40.csv', userno:int=30, fakeno:int=15, recno:int=10,  split:str='train', train_split_frac:float=.7, authentics:bool=True, impostors:bool=True, rand_state:int=0, transform=None):
        """ Initializes the events dataset. Builds custom dataframe according to userno, fakeno and recno

        Args:
            csv_file (string): Path to the csv file with all annotations.            
            userno (int, optional): _description_. Defaults to 30.
            fakeno (int, optional): _description_. Defaults to 15.
            recno (int, optional): _description_. Defaults to 10.
            split (str, optional): Train or test split. Defaults to 'train'.
            train_split_frac (float, optional): Train split fraction. Defaults to .7.
            authentics (bool, optional): Include authentics in the dataset. Defaults to True.
            impostors (bool, optional): Include impostors in the dataset. Defaults to True.
            rand_state (int, optional): Seed. Defaults to 0.
            transform (callable, optional): Optional transform to be applied on a sample.        
        """
        self.csv_file = csv_file
        self.userno = userno
        self.fakeno = fakeno
        self.recno = recno
        self.split = split
        self.train_split_frac = train_split_frac
        self.authentics = authentics
        self.impostors = impostors
        self.rand_state = rand_state

        self.transform = transform

        self.user_mapping = map_users(list(range(self.userno)))

        assert isinstance(split, str) and split in ['train', 'test'], f'Split must be either <train> or <test>. Got <{split}> ({type(split)})'
        self._assert(csv_file, userno, fakeno, recno, train_split_frac, authentics, impostors)

        self.df = self.get_df(split)

    def _assert(self, csv_file, userno, fakeno, recno, train_split_frac, authentics, impostors):
        assert os.path.isfile(csv_file), f'File <{csv_file}> does not exist'
        assert 1<= userno <=76, f'Lists must contain at least 1 user and at most 76 users. Got userno={userno}'
        assert fakeno < userno, f'Fake users number [fakeno] ({fakeno}) must be inferior to Real users number [userno] ({userno})'
        assert 1<= recno <=10, f'Real recordings [recno] must be in range (1,10). Got recno={recno}'
        assert 0. < train_split_frac <1., f'Split fraction must be in range (0., 1.). Got train_split_frac={train_split_frac}'
        assert authentics or impostors, f'Lists must contain authentics, impostors or both. Got authentics={authentics}, impostors={impostors}'
 
    def get_df(self, split:str) -> pd.DataFrame:
        """ Get SynFED dataframe

        Args:
            split (str):  train or test split

        Returns:
            pd.DataFrame: split dataframe
        """
        np.random.seed(self.rand_state)

        users = range(self.userno)

        real_users_train = np.ones((self.userno,self.recno), dtype=int) * np.arange(self.recno)
        fake_users_train = np.array( [np.random.choice(list(set(users)-set([u])) , self.fakeno, replace=False) for u in users] ) # Random choice fake user * fakeno  for each real user for train

        df_all = pd.read_csv(self.csv_file, sep=';')
            
        df_all_train = df_all.groupby(['path','rec','user']).sample(frac=self.train_split_frac, replace=False, random_state=self.rand_state) ## Split normalized 

        df_all_train.sort_values(by=['user', 'rec', 'path', 'frameno_list'], ascending=True, inplace=True)
        df_all_test = df_all.drop(df_all_train.index)
        df_all_test.sort_values(by=['user', 'rec', 'path', 'frameno_list'], ascending=True, inplace=True)

        df_train = pd.DataFrame(columns=df_all.columns)
        df_test = pd.DataFrame(columns=df_all.columns)

        def append_samples2df(samples, df):
            return pd.concat([df, samples], ignore_index=True)
            
        for dynamic in users:
            # TRAIN
            if self.authentics:
                df_train = append_samples2df(df_all_train[ (df_all_train['folder']=='s3dfm') & (df_all_train['user']==dynamic) & (df_all_train['rec'].isin(real_users_train[dynamic])) ], df_train)
            
            if self.impostors:
                df_train = append_samples2df(df_all_train[ (df_all_train['folder']=='deepfakes_v1') & (df_all_train['rec']==dynamic) & (df_all_train['user'].isin(fake_users_train[dynamic])) ], df_train)
                
        ## TEST
        if self.authentics:
            df_test = append_samples2df(df_all_test[ (df_all_test['folder']=='s3dfm') & (df_all_test['user'].isin( list(users)) ) & (df_all_test['rec'].isin( list(users) ) )] , df_test) # ALL REAL
        if self.impostors:
            df_test = append_samples2df(df_all_test[ (df_all_test['folder']=='deepfakes_v1') & (df_all_test['user'].isin( list(users)) ) & (df_all_test['rec'].isin( list(users) ) )] , df_test) # ALL FAKES

        return {'train':df_train, 'test':df_test}[split]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.df.loc[idx]  # pd.Dataframe line -> path, frameno_list, user, rec
                                 #                      DATA/event_frames/{line.user}/{line.rec}/frame_{line.user}{line.rec}_{frameno:0>4}, frameno_list, user, rec

        idxs = np.array(line.frameno_list.strip("[ ]").split(","), dtype=int)  # converts string "[ 1, 2, 3, 4, 5 ]" to np.array([1,2,3,4,5])
        
        # Get subclip
        clip = np.load(line.path)  # shape: (no_channels, frames, H, W)
        subclip = clip[:, idxs, ...]
        
        # Get label
        label = self.user_mapping[line.user] if line.real else self.user_mapping[line.rec] 
        
        if self.transform :
            subclip = self.transform(subclip)
            
        sample = {
            "clip": subclip,
            "label": label,
            "ohlabel": np.eye(self.userno)[label],
            "user": line.user,
            'task':line.get('task', 'THIS LIST DOES NOT CONTAIN TASK COLUMNS'), # SynFED does not have tasks
            "recording": line.rec,
            'folder': line.get('folder', find_term_in_string(string=line.path)), # NVFSD does not have folder
            'real':line.real,
        }
        

        return sample


    def __len__(self):
        return len(self.df)


    def __repr__(self) -> str:
        return f"""Events Dataset
  csv file: {self.csv_file}

  userno: {self.userno}
  fakeno: {self.fakeno}
  recno: {self.recno}
  user mapping: {self.user_mapping}

  split: {self.split}
  split_frac: {self.train_split_frac if self.split=='train' else 1-self.train_split_frac}

  authentics: {self.authentics}
  impostors: {self.impostors}

  rand_state: {self.rand_state}
  
  dataset length: {len(self)}
"""
