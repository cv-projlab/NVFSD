import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from ipywidgets import Layout, VBox, interactive, widgets


def create_double_selection(_interact_func_, options_dict: Dict, display_flag:bool=False, output=None) -> Tuple:
    """ Creates double selection with a dependent selection

    Args:
        interact_func (_type_): _description_
        options_dict (Dict): _description_
        display_flag (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[widgets.Selection, widgets.Selection]: _description_
    """
    select1 = widgets.Select(options=options_dict.keys())
    init = select1.value
    select2 = widgets.Select(options=options_dict[init])

    def select_func(aux_name):
        select2.options = options_dict[aux_name]
        
    interactive_element1 = widgets.interactive(select_func, aux_name=select1)
    if display_flag:
        display(interactive_element1, output) if output else display(interactive_element1)

    interactive_element2 = interactive(_interact_func_, name=select2)
    interactive_element2.children[0].layout = Layout(width='1500px')
    if display_flag:
        display(interactive_element2, output) if output else display(interactive_element2)

    return interactive_element1, interactive_element2
    

def create_selection(_interact_func_, options_list: List, display_flag:bool=False, output=None) -> Tuple:
    """ Creates double selection with a dependent selection

    Args:
        interact_func (_type_): _description_
        options_dict (Dict): _description_
        display_flag (bool, optional): _description_. Defaults to False.

    Returns:
        Tuple[widgets.Selection, widgets.Selection]: _description_
    """
    select1 = widgets.Select(options=options_list)
    interactive_element1 = widgets.interactive(_interact_func_, predictions=select1)
    
    if display_flag:
        display(interactive_element1, output) if output else display(interactive_element1)

    return interactive_element1


def create_button(_interact_func_, name:str = 'Button', display_flag:bool=False, output=None) -> widgets.Button:
    interactive_element = widgets.Button(
        description=name,
        disabled=False,
        button_style='success', # 'success', 'info', 'warning', 'danger' or ''
    )
    if display_flag:
        clear_output()
        display(interactive_element, output) if output else display(interactive_element) 
        
    interactive_element.on_click(_interact_func_)
    
    return interactive_element

def create_checkbox(_interact_func_, name:str = 'CheckBox', display_flag:bool=False, output=None, **kwargs) -> widgets.Checkbox:
    interactive_element = widgets.Checkbox(
        description=name, **kwargs
    )
    
    if display_flag:
        clear_output()
        display(interactive_element, output) if output else display(interactive_element)
        
    interactive_element.observe(_interact_func_)
    
    return interactive_element

def create_slider(_interact_func_, display_flag:bool=False, output=None, **kwargs) -> widgets.IntSlider:
    """ Create slider with callback

    Args:
        _interact_func_ (_type_): _description_
        name (str, optional): _description_. Defaults to 'Slider'.
        display_flag (bool, optional): _description_. Defaults to False.
        output (_type_, optional): _description_. Defaults to None.
        kwargs : min, max, step

    Returns:
        widgets.Button: _description_
    """
    interactive_element = widgets.IntSlider(**kwargs)
    
    if display_flag:
        clear_output()
        display(interactive_element, output) if output else display(interactive_element)
        
    interactive_element.observe(_interact_func_)

    return interactive_element

def create_vbox(*args):
    return widgets.VBox([arg for arg in args])

def create_hbox(*args):
    return widgets.HBox([arg for arg in args])

def get_widgets_output():
    return widgets.Output()


if __name__ == '__main__':
    
    pass

