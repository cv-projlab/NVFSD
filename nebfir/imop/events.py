# TODO:
#       - add save frame during convertion
#
import os
from pathlib import Path
import time
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd

from ..tools.tools_basic import tprint
from ..tools.tools_visualization import show_img_cv2



def sort_events(events_file: str, start_0: bool = False):
    """ start_0: timestamps start at 0"""
    pc = pd.read_csv(events_file, sep=",")

    pc.columns = ["timestamp", "x", "y"]  # pc.columns = ["frame", "timestamp", "x", "y"]
    # Sort by timestamp
    pc.sort_values(by="timestamp", inplace=True, ignore_index=True)
    # Offset the time axis to start at zero
    if not start_0:
        pc.loc[:, "timestamp"] = pc.loc[:, "timestamp"] - pc.loc[0, "timestamp"]
    # Find the total number of frames
    last_row = pc.tail(1)["timestamp"].values[0]

    new_arr = pc.to_numpy()

    return new_arr, last_row


def create_ts_csv(path_to_events: str, ts_s: bool = True):
    """ Creates the events file from events.txt
        
    Args:
        path_to_events (str): path to events.txt file. events.txt has 4 columns: [timestamps, x, y, polarity] 
        ts_s (bool, optional): timestamp is in seconds. Defaults to True.
    """
    path_to_events = Path(path_to_events)
    if path_to_events.is_dir():
        path_to_events = path_to_events / "events.txt"

    df = pd.read_csv(str(path_to_events), sep=" ", skiprows=5)
    df.columns = ["timestamp", "x", "y", "p"]  # "timestamp", "x", "y", "polarity"
    if ts_s:
        df["timestamp"] = (df["timestamp"] * 1_000_000).astype(int)  # micro seconds
    df = df.drop(columns=["p"])  # Remove polarity

    df.to_csv(f"{path_to_events.parent}/pc.csv", sep=",", index=None)


class BaseEventsRepresentation:
    """ Events Representation Base Class
    
        NOTE: tqdm integration available
        """

    DEBUG = False

    def __init__(self, frame_shape: Tuple = (800, 1280), dT: int = 40000, x_crop_size: int = 800) -> None:
        self.frame_shape: Tuple[int, int] = frame_shape
        self.dT: int = dT  # 40ms
        self.x_crop_size = x_crop_size  # 800

        self._init_extra()

    def _init_extra(self):
        """ Initialize extra values for different representations """
        raise NotImplementedError(f"_init_extra() method is not implemented for {self.__class__.__name__} class")

    def reset(self, arr: np.ndarray, bar: tqdm = None) -> None:
        last_ts = arr[-1][0]  # arr[last_row][timestamp]     arr_shape: [[timestamp x y], ...]
        FRAME_NO = round(last_ts / self.dT)

        self.FRAME_NO = FRAME_NO
        self.FRAMES = np.zeros((*self.frame_shape, FRAME_NO), dtype=np.uint8)

        if bar is not None:
            bar.total = FRAME_NO
            bar.reset()

        self._init_extra()

    def convert_events_to_frames(self, arr: np.ndarray, bar: tqdm = None) -> np.ndarray:
        """ Convert events to frames

        Args:
            arr (np.ndarray): array of [timestamps, x, y]
            bar (tqdm, optional): tqdm progress bar. Defaults to None.

        Returns:
            np.ndarray: Frames clip

        Raises:
            NotImplementedError: if subclass does not have this method implemented

        """

        if (self.convert_events_to_frames.__qualname__).split(".")[0] != f"{BaseEventsRepresentation.__class__.__name__}":
            self.reset(arr, bar)
            return

        raise NotImplementedError(
            f"convert_events_to_frames() method is not implemented for {self.__class__.__name__} representation\n\t\tDont forget to add 'super().convert_events_to_frames(arr=arr, bar=bar)' at the start of the method"
        )

    def get_frames(self) -> np.ndarray:
        """ Get the Frames clip

        Returns:
            np.ndarray: Frames clip
        """
        return self.FRAMES

    def get_cropped_frames(self, x: int):
        CROPPED_FRAMES = self.FRAMES[:, x - 1 : x - 1 + self.x_crop_size, :]
        return CROPPED_FRAMES

    def __repr__(self) -> str:
        return f"Events Representation: {self.__class__.__name__}\nFrame Shape: {self.frame_shape}\nIntegration Time dT (micro-seconds): {self.dT}"

    def toggle_debug_mode(self):
        self.DEBUG = not self.DEBUG


class AETS(BaseEventsRepresentation):
    """Accumulated Exponential Time Surface"""

    # AETS Parameters
    tau: int = 0.1 * 1e6

    def _init_extra(self):
        self.sigma: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)  # S
        self.gama: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)  # P
        self.rho: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)  # G

    def convert_events_to_frames(self, arr: np.ndarray, bar: tqdm = None) -> np.ndarray:
        super().convert_events_to_frames(arr=arr, bar=bar)

        framek_1no = 0
        tk_1 = 0
        for tk, xk, yk in arr:

            # frame number of the k-th event
            framekno = np.floor(tk / self.dT)

            # if framekno changed aka new frame
            if framekno != framek_1no:

                self.gama = (self.gama + self.rho) * np.exp((self.sigma - tk_1) / self.tau)
                normgama = self.gama / np.abs(np.max(self.gama.flatten(order="F")))

                self.FRAMES[:, :, int(framek_1no)] = (255 * normgama).astype("uint8")

                if self.DEBUG:
                    show_img_cv2(self.FRAMES[:, :, int(framek_1no)], wait_time=20)
                    tqdm.write(f"gama: {np.amax(self.gama)}")

                framek_1no = framekno

                if bar is not None:
                    bar.set_description_str(f"PNG frame: {int(framekno):0>4}")
                    bar.update()

            # Update S and P
            self.sigma[yk, xk] = tk
            self.rho[yk, xk] = 1

            tk_1 = tk

        return self.FRAMES


class SNN(BaseEventsRepresentation):
    # SNN Parameters

    threshold: float = 1.2
    decay: float = 0.0015
    margin: int = 1
    spikeVal: int = 1

    def _init_extra(self):
        self.network: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)  # S
        self.timenet: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)  # P
        self.firing: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)  # G

    def convert_events_to_frames(self, arr: np.ndarray, bar: tqdm = None):
        super().convert_events_to_frames(arr=arr, bar=bar)

        framek_1no = 0
        for tk, xk, yk in arr:
            # frame number of the k-th event
            framekno = np.floor(tk / self.dT)

            # if framekno changed aka new frame
            if framekno != framek_1no:
                if framek_1no > 0:
                    self.FRAMES[:, :, int(framek_1no)] = (255 * 2 * (1.0 / (1 + np.exp(-self.firing)) - 0.5)).astype("uint8")

                self.firing = np.zeros_like(self.firing)
                self.timenet = np.ones_like(self.timenet) * tk

                if self.DEBUG:
                    show_img_cv2(self.FRAMES[:, :, int(framek_1no)], wait_time=20)

                if bar is not None:
                    bar.set_description_str(f"PNG frame: {int(framekno):0>4}")
                    bar.update()

            if framekno <= self.FRAME_NO:

                escape_time = (tk - self.timenet[yk, xk]) / 1000
                residual = self.network[yk, xk] - self.decay * escape_time

                self.network[yk, xk] = residual + escape_time
                self.timenet[yk, xk] = tk

                if self.network[yk, xk] > self.threshold:
                    self.firing[yk, xk] = 1

                    for m in range(-self.margin, self.margin + 1):
                        for n in range(-self.margin, self.margin + 1):
                            if xk + m > 0 and xk + m <= self.frame_shape[0] and yk + n > 0 and yk + n <= self.frame_shape[1]:
                                self.network[yk + n, xk + m] = 0

            framek_1no = framekno

        return self.FRAMES


class FRQ(BaseEventsRepresentation):
    """Event frequency representation"""

    # FRQ Parameters

    def _init_extra(self):
        self.counter: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)

    def convert_events_to_frames(self, arr: np.ndarray, bar: tqdm = None):
        super().convert_events_to_frames(arr=arr, bar=bar)

        framek_1no = 0
        for tk, xk, yk in arr:
            # frame number of the k-th event
            framekno = np.floor(tk / self.dT)

            # if framekno changed aka new frame
            if framekno != framek_1no:
                if framek_1no > 0:
                    self.FRAMES[:, :, int(framek_1no)] = (255 * 2 * (1.0 / (1 + np.exp(-self.counter)) - 0.5)).astype("uint8")

                self.counter = np.zeros_like(self.counter)

                if self.DEBUG:
                    show_img_cv2(self.FRAMES[:, :, int(framek_1no)], wait_time=1)

                if bar is not None:
                    bar.set_description_str(f"PNG frame: {int(framekno):0>4}")
                    bar.update()

            if framekno <= self.FRAME_NO:
                self.counter[yk, xk] = self.counter[yk, xk] + 1

            framek_1no = framekno

        return self.FRAMES


class TBR(BaseEventsRepresentation):
    # TBR Parameters

    BIN_NO = 8  #% Number of bins %% max pixel value= 255

    def _init_extra(self):
        self.sigma: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)  # S
        self.gama: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)  # P
        self.rho: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)  # G


class SAE(BaseEventsRepresentation):
    # SAE Parameters

    start_time = 0

    def _init_extra(self):
        self.timestamp: np.ndarray = np.zeros(self.frame_shape, dtype=np.float64)


class LeakySurface(BaseEventsRepresentation):
    pass




def main():
    events_rep = AETS(dT=40_000, frame_shape=(600, 600))  # AETS SNN FRQ TBR SAE LeakySurface
    print(events_rep)
    events_rep.toggle_debug_mode()
    ################ EVENTS CONVERSION ################
    size = 50000
    np.random.seed(0)

    # events, _ = sort_events("some_events_file", start_0=False)  ## Load events
    # events, _ = sort_events_simao("DATA/carros.csv", start_0=False)  ## Load events
    # events, _ = sort_events("DATA/events/s3dfm/p1/s1/pc.csv", start_0=False)  ## Load events
    # events, _ = sort_events("DATA/events/deepfakes/src1/dst1/pc.csv", start_0=False)  ## Load events
    events, _ = sort_events("/mnt/DATADISK/Datasets/face/SynFED/Events/deepfakes_v1/u0/r0/pc.csv", start_0=False)  ## Load events
    # events = np.hstack( (np.arange(size).reshape(size,1), np.random.randint(0,600, size).reshape(size,1), np.random.randint(0,600, size ).reshape(size,1) ) ) # 3 cols for [timestamp, x ,y]

    print(events.shape)
    print(events[:10])

    # events_rep.activate_debug_mode()

    events_rep.convert_events_to_frames(arr=events, bar=tqdm(range(events.shape[0])))
    # events_rep.convert_events_to_frames(arr=events)

    # for i in tqdm(range(events_rep.FRAMES.shape[2])):
    #     show_img_cv2(events_rep.FRAMES[..., i], wait_time=20)

if __name__ == "__main__":
    main()
