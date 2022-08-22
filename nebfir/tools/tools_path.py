from itertools import islice

from nebfir.env import *

from .tools_basic import dict2str

ROOT_PATHS = 'data/PATHS.json'

########################################## PATHS ##########################################
def join_paths(*args: str):
    return os.path.join(*args)


def exists(path: str):
    return os.path.exists(path)


def is_file(path: str):
    return os.path.isfile(path)


def is_dir(path: str):
    return os.path.isdir(path)




def get_path(level: Tuple[str]) -> str:
    """
        Args:
            
            level (str, optional): Paths options: datasets, models, . Defaults to 'datasets'.

        Returns:
            [type]: path, newpath
        """

    assert os.path.isfile(ROOT_PATHS)
    with open(ROOT_PATHS, 'r') as f:
        paths = json.load(f)

    if isinstance(level, tuple) or isinstance(level, list):
        path = paths
        for l in level:
            l = l.lower() 
            path = path.get(l)
    elif isinstance(level, str):
        path = paths.get(level)
    else:
        raise AttributeError('level attribute must be a tuple with dictionary keys to read from PATHS.json file')
    
    assert isinstance(path, str), f'Resulting path: {path} , is not of type str. Make sure the last value in the level tuple is the last key for the PATHS.json file'

    return path


def get_multiple_paths(levels: Union[Tuple[str], Tuple[Tuple], str]) -> Union[Tuple[str], str]:
    if (isinstance(levels, tuple) or isinstance(levels, list)) and (isinstance(levels[0], tuple) or isinstance(levels[0], list)):
        return [get_path(level=tup) for tup in levels] 
    elif isinstance(levels, str) or ( isinstance(levels, tuple) or isinstance(levels, list) and isinstance(levels[0], str) ):
        return get_path(level=levels)
    else:
        raise TypeError('Wrong type for level attribute. Must be Tuple[Tuple[str]] or Tuple[str] or str')



def create_paths(working_dir=None):
    if working_dir is None:
        working_dir = str(Path.cwd())
        
    pdict = {'runs':join_paths(working_dir, "data/runs/"), 
            "lists": join_paths(working_dir, "data/inp/lists/"),           
            "xlocation": join_paths(working_dir, "data/inp/xlocation_40users.mat"),
            
            }
    
    with open(join_paths(working_dir,ROOT_PATHS), 'w') as jfile:
        json.dump(pdict, jfile, indent=4)

# def create_paths(working_dir=None):
#     if working_dir is None:
#         working_dir = str(Path.cwd())
        
#     pdict = {
#             "model": {
#                     "weights": join_paths(working_dir, "data/out/weights/"),
#                     "logs": join_paths(working_dir, "data/out/model_logs/"),
#                     "tests":join_paths(working_dir, "data/out/tests/"),
#                     },
            
#             "dry-run": {
#                     "weights": join_paths(working_dir, "data/out/dry_run/weights/"),
#                     "logs": join_paths(working_dir, "data/out/dry_run/model_logs/"),
#                     "tests":join_paths(working_dir, "data/out/dry_run/tests/"),
#                     },
            
#             "lists": join_paths(working_dir, "data/inp/lists/"),           
#             "xlocation": join_paths(working_dir, "data/inp/xlocation_40users.mat"),
            
#             }
    
#     with open(join_paths(working_dir,ROOT_PATHS), 'w') as jfile:
#         json.dump(pdict, jfile, indent=4)

def print_paths():
    with open(ROOT_PATHS, 'r') as jfile:
        paths = json.load(jfile)
    print(dict2str(paths, indentation=4))
    
    




def create_mid_dirs(path: str):
    """ Creates directories of the given path if its a file or a directory

    Args:
        path (str): path
    """
    if Path(path).suffix:
        path = Path(path).parent

    if not is_dir(path):
        os.makedirs(path)


def get_empty_folders_list(folder_paths_list: Union[List[str], str]) -> List[str]:
    """ Return the empty folders paths

    Args:
        folder_paths_list (Union[List[str], str]): can be a string for the base directory or a list with the folder paths

    Returns:
        List[str]: empty folder paths list
    """
    if isinstance(folder_paths_list, str):
        folder_paths_list = sorted(glob(os.path.join(folder_paths_list, "**", "*/"), recursive=True))

    return [folder for folder in folder_paths_list if len(os.listdir(folder)) == 0]


def remove_folders(folders):
    [os.rmdir(dir_) for dir_ in folders]

def remove_empty_folders(folder_paths_list: Union[List[str], str]):
    """ Removes empty folders
    Args:
        folder_paths_list (Union[List[str], str]): can be a string for the base directory or a list with the folder paths
    """
    empty_folders = get_empty_folders_list(folder_paths_list)
    remove_folders(empty_folders)
    print('Removed empty folders!')


def remove_files(files: Union[str, Path, List[str], List[Path]]) -> None:
    """ Removes a list of files

    Args:
        files (Union[str, Path, List[str], List[Path]]): List of files. NOTE: < files > can be a single file if type is str or pathlib.Path

    Raises:
        TypeError: Files must be of type str or pathlib.Path
    """
    if isinstance(files, (str, Path)): files = [files]

    for file in files:
        print(f'Removing file < {file} > with type {type(file)}')
        try:
            if isinstance(file, str): os.remove(file)
            elif isinstance(file, Path): file.unlink()
            else: raise TypeError(f'Wrong file type. Expected type {str, Path}. Got {type(file)}')

        except Exception as e :
            print(e)


def tree(dir_path: Path, level: int=-1, limit_to_directories: bool=False, length_limit: int=1000, filter_list: List=None, sort_key=None):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    
    Grabbed from stackoverflow.
    
    link: https://stackoverflow.com/a/59109706
    """    
    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    tee =    '├── '
    last =   '└── '

    if filter_list is None:
        filter_list = []

    """Given a directory Path object print a visual tree structure"""
    dir_path = Path(dir_path) # accept string coerceable to Path
    files = 0
    directories = 0
    def inner(dir_path: Path, prefix: str='', level=-1, filter_list=[]):
        nonlocal files, directories
        if not level: 
            return # 0, stop iterating
        if limit_to_directories:
            contents = sorted([d for d in dir_path.iterdir() if d.is_dir()], key=sort_key)
        else: 
            contents = sorted(list(dir_path.iterdir()), key=sort_key)
            
        contents = list(filter(lambda x: None if x.name in filter_list else x, contents))

        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space 
                yield from inner(path, prefix=prefix+extension, level=level-1, filter_list=filter_list)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1
                
    print(dir_path.name)
    iterator = inner(dir_path, level=level, filter_list=filter_list)
    for line in islice(iterator, length_limit):
        print(line)
    if next(iterator, None):
        print(f'... length_limit, {length_limit}, reached, counted:')
    print(f'\n{directories} directories' + (f', {files} files' if files else ''))
    
    
    

def clean_logs():
    basedir = 'data/out/model_logs'

    basename = lambda x: Path(x).stem

    keep = list(map(basename , glob(f'{basedir}/*.txt')))

    discard_filter = lambda x: x if not any([keep_ in Path(x).stem for keep_ in keep]) else ''
    keep_filter = lambda x: x if any([keep_ in Path(x).stem for keep_ in keep]) else ''

    discarded_files = sorted(filter(discard_filter , glob(f'{basedir}/*.*')))
    keep_files = sorted(filter(keep_filter , glob(f'{basedir}/*.*')))

    print('# of discarded_files : ', len(discarded_files))
    print('# of keep_files : ', len(keep_files))

    assert not all([kfile in discarded_files for kfile in keep_files]), f'Something went wrong. < discarded_files > cant contain any of the keep_files !'

    remove_files(discarded_files)
