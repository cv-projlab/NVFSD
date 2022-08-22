from typing import Any

from ..env import *


########################################## Random ##########################################
class Group:
    """ Groups multiple class instances and use their methods with a single call"""
    def _apply(self, item:Any, method:Union[str,Any], *args, **kwargs) -> Any:
        """ Applies a method to the item

        Args:
            item (Any): class object
            method (Union[str,Any]):  If method is of type 'str', the method is a class instance method

        Returns:
            Any:  method output
        """
        if isinstance(method, str):
            return getattr(item, method)(*args, **kwargs) # The method is a class instance method
        
        return method(item, *args, **kwargs) # The method is not a class instance method

    def apply(self, method, *args, **kwargs):
        """ Apply the method to each item in the group

        Args:
            method : class method to apply

        Raises:
            NotImplementedError: Implement this method to apply the method with different group types
        """
        raise NotImplementedError()


class GroupAsList(Group):
    """ Groups items as a list """
    def __init__(self, group:List=[], *group_items) -> None:
        self.group = group + [*group_items]

    def apply(self, method, *args, **kwargs) -> List:
        group_out = [self._apply(item, method, *args, **kwargs) for item in self.group]

        return group_out

    def __repr__(self) -> str:
        return str(self.group)


class GroupAsDict(Group):
    """ Groups items as a dictionary """
    def __init__(self, group:Dict={}, **group_items:Any) -> None:
        """ Initializes Dictionary group

        Args:
            group (dict, optional): group dict. Defaults to {}.
            **group_items (Any): 

        """
        assert isinstance(group, dict)

        self.group = group
        self.group.update(dict(**group_items))

    def apply(self, method, *args, **kwargs) -> Dict:
        group_out = {key: self._apply(item, method, *args, **kwargs) for key, item in self.group.items()}

        return group_out

    def __repr__(self) -> str:
        return dict2str(self.group, level=0, indentation=2)


########################################## Random ##########################################

def isfloat(x:str) -> bool:
    return x.replace('.','',1).isnumeric()



########################################## PRINT ##########################################
def tprint(s: str):
    tqdm.write(s)


def dict2str(_dict: Dict, level: int = 0, indentation: int = 4) -> str:
    """ Converts a dictionary to a string

        Args:
            _dict (Dict): a dictionary to convert
            level (int, optional): Defaults to 1.
            indentation (int, optional): Defaults to 4.

        Raises:
            TypeError: Dictionary must contain values of either type: tuple[int or str], list[int or str], dict, int, str

        Returns:
            _str (str): dictionary as a string

        TODO:
            [ ] check for other types inside lists/tuples
        """

    _str = ""
    indent = level * indentation * " "

    for k, v in _dict.items():
        if isinstance(v, (str, int, float)):
            _str += indent + f"{k}: "  # KEY
            _str += f"{str(v)}\n"  # VALUE

        elif isinstance(v, (tuple, list)):
            _str += indent + f"{k}: "  # KEY
            _str += f'{", ".join([str(v0) for v0 in v])}\n'  # VALUE

        elif isinstance(v, dict):
            _str += indent + f"{k}: \n"  # KEY
            _str += dict2str(v, level=level + 1, indentation=indentation)  # VALUE
        else:
            raise TypeError(f"Dictionary must contain values of either type: tuple[int or str], list[int or str], dict, int, str . Unsupported value: {v}")

    return _str


def update_nested_dict(old: Dict, new: Dict) -> Dict:
    """ Updates nested dict

    Args:
        old (Dict): base dict to update
        new (Dict): new dict

    Returns:
        Dict: Updated base dict
    """
    for k, v in new.items():
        if isinstance(v, dict):
            old[k] = update_nested_dict(old.get(k, {}), v) 
        else:
            if k in ['name'] and v != old[k]: # If using different modules, use all the new parameters ## and k in old 
                return new
            if k in ['description']: # Extend description
                old[k] += v
            else:
                old[k] = v

    return old


def key_sort_by_numbers(x: str, split_by:str='/') -> List[int]:  # Example: x = "DATA/images/s3dfm/u0/r0"
    """ Key function to use with sorted(..., key=sort_by_numbers)

    Args:
        x (str): path. Example: x = "DATA/images/s3dfm/u0/r0"

    Returns:
        List[int]: list with integers to sort folders or files
    """
    split_path = x.split(split_by)  # ["DATA", "images", "s3dfm", "u0", "r0"]
    numbers_list_without_letters_str = [x_.strip(string.ascii_letters + string.punctuation) for x_ in split_path]  # ["", "", "3", "0", "0"]
    numbers_list_without_spaces_str = " ".join(numbers_list_without_letters_str).split()  # ["3", "0", "0"]
    numbers_list_int = list(map(int, numbers_list_without_spaces_str))  # [3, 0, 0]

    return numbers_list_int


class Mode(Enum):
    big = auto()
    small = auto()
    
def filter_string(x: str, comparing_tuple: Tuple = (10,10), flag=Mode.small):  # Example: x = "DATA/images/s3dfm/u0/r0"
    """_summary_

    Args:
        x (str): _description_
        comparing_tuple (Tuple, optional): compare result numbers with a filter tuple. Defaults to (10,10).
        flag (_type_, optional): _description_. Defaults to Mode.small.

    Returns:
        _type_: _description_
    """
    
    split_path = x.split("/")  # ex: ["DATA", "images", "s3dfm", "u0", "r0"]
    numbers_list_without_letters_str = [x_.strip(string.ascii_letters + string.punctuation) for x_ in split_path]  # ["", "", "3", "0", "0"]
    numbers_list_without_spaces_str = " ".join(numbers_list_without_letters_str).split()  # ["3", "0", "0"]
    numbers_list_int = list(map(int, numbers_list_without_spaces_str))  # [3, 0, 0]
    
    tup = tuple(numbers_list_int[~1:]) # (0, 0)
    
    result = all(x < y if flag==Mode.small else x >= y for x, y in zip(tup, comparing_tuple)) # comparing_tuple=(10,10) -> (0 < 10 and 0 < 10) = True
    return result



def choose_model(model_dir: str, model_name: str):
    model_paths = sorted(glob(os.path.join(model_dir, f"{model_name}*.pth")))

    if not model_paths:
        raise FileNotFoundError(f"There are no pretrained models with name: {model_name}")

    key_words = ["best"]  # , 'min', 'max'
    matching = match_keywords(model_paths, key_words)
    if not matching:
        return model_paths[-1]

    return matching[-1]


def match_keywords(_list: List[str], key_words: List[str]) -> List[str]:
    matching = []
    for kw in key_words:
        matching.extend([s for s in _list if kw in s])

    return matching


def find_term_in_string(string :str, search_terms_list: List[str]=None) -> str:
    if not search_terms_list:
        search_terms_list = ['s3dfm', 'deepfakes_v1', 'deepfakes_v2', 'deepfakes_v3']
    
    loc = dict(zip(search_terms_list, [None]*len(search_terms_list)))
    for s in search_terms_list:
        loc[s] = string.find(s)
        
    k = [k for k,v in loc.items() if v > -1 ] # return "_NOT_FOUND_" if not found else return the key term
    
    assert len(k) <=1, f'Multiple terms {k} found in the search term {string}'
    
    return k[0] if k else '_NOT_FOUND_'


def get_val_from_string(string: str, keyword: str, val_length:int=1, strip_char:str='_') -> str:
    """ Search a string to find a keyword and return the value. If '_' in the string it is striped

    Args:
        string (str): _description_
        keyword (str): _description_
        val_length (int, optional): _description_. Defaults to 1.

    Returns:
        str: _description_
    """
    kw_length = len(keyword)
    idx = string.index(keyword) + kw_length
    return string[idx:idx+val_length].strip(strip_char)
    
def multi_split(str_:str, separators:Union[List[str], Tuple[str], str], map_func=None) -> List[str]:
    """ Splits a string based on the separators

    NOTE: Removes empty strings after spliting

    Args:
        str_ (str): String to split
        separators (Union[List[str], Tuple[str], str]): List of string separators. Compatible with module string, ex: separators = string.punctuation + string.whitespace

    Returns:
        List[str]: List of strings split with the separators
    """
    assert isinstance(str_, str), f'Input str_ must be of type str. Got type {type(str_)}'
    assert isinstance(separators, (list, tuple, str)), f'Separators must be of type List, Tuple or str. Got type {type(separators)}'
    last_split_w_empty_filter = lambda str__, sep: list(filter(lambda x: x!='', str__.split(sep)))

    if len(separators)<1:
        return [str_]
    if len(separators)==1:
        out = last_split_w_empty_filter(str_, separators[0])

        return map_func(out) if map_func is not None else out


    old_sep = separators[0]
    for sep in separators:
        str_ = str_.replace(old_sep, sep)
        old_sep = sep

    out = last_split_w_empty_filter(str_, old_sep)
    
    return map_func(out) if map_func is not None else out

separators_letters_punctuation = string.ascii_letters+string.punctuation
separators_punctuation = string.punctuation
separators_numbers_punctuation = string.digits+string.punctuation

str2int_list = lambda x: [int(x_) for x_ in x]


if __name__ == "__main__":
    # my_dict = {"a": {"b": [1, {2: "owiefnc"}, 3, 4, 5, 6], "c": 2, "d": {1: 3, 4: 6}}, "e": 999}
    # print(dict2str(my_dict, indentation=1))
    # 
    # 
    # print(choose_model(model_dir=get_path(level=('model', 'pretrained')), model_name='model_*_EXP*_DS*_stride1_nClasses*_Dur500_AETS_40ms'))
    # match = match_keywords(_list=['this is a little comment' ], key_words=['a_test', 'something','little','comment'])
    # match = match_keywords(_list=['this','is','a','little','comment' ], key_words=['a_test', 'something','little','comment'])
    # print(match)


    pass

