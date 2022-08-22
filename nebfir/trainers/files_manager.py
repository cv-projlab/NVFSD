from copy import deepcopy
from ..config.configurations import Configurations
from ..env import *
from ..tools.tools_basic import dict2str, update_nested_dict
from ..tools.tools_path import get_multiple_paths


class Logger:
    """ 

    Returns:
        _type_: _description_
    """
    formatter = logging.Formatter(fmt="%(asctime)s : %(levelname)s : %(message)s",
                                    datefmt="%d/%m/%Y %H:%M:%S")

    def setup_logger(self, name, log_file, console_log=False, level=logging.INFO):
        """To setup as many loggers as you want"""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        handler_list = [
            logging.FileHandler(log_file, mode="w") ,
            logging.StreamHandler() # console stream
        ] if console_log else [logging.FileHandler(log_file, mode="w")]
        
        for handler in handler_list:
            handler.setFormatter(self.formatter)
            logger.addHandler(handler)

        return logger

    def remove_console_logger_handler(self, name):
        logger = logging.getLogger(name)  # root logger
        for hdlr in logger.handlers:  # remove all old handlers
            if not isinstance(hdlr, logging.FileHandler):
                logger.removeHandler(hdlr)

    def add_console_logger_handler(self, name):
        logger = logging.getLogger(name)  # root logger
        handler = logging.StreamHandler()
        handler.setFormatter(self.formatter)
        logger.addHandler(handler)





class NetConfig:
    def __init__(self, file_list, args):
        self.cfg_name = deepcopy(file_list)
        self.config = Configurations.load_multiple_cfg(cfg_list=file_list)
        

        self.USER_NO = self.config["trainer"]["model"]['num_classes']
        self.EPOCHS = args.epochs or self.config["trainer"]["epochs"]
        self.BATCHSIZE = args.batch_size or self.config["trainer"]["batch_size"]

        self.dT = self.config["data"]['dataset']["dT"]  # Frame Time: 40 ms
        self.subclipT = self.config["data"]['dataset']["subclipT"]  # Learnable Video Duration: 500 ms
        self.stride = self.config["data"]['dataset']["stride"]

        self.data_type, self.reppath = self.config["data"]['dataset']["type"].split('_') if len(self.config["data"]['dataset']["type"].split('_')) > 1 else (self.config["data"]['dataset']["type"], '')
        
        self.train_list_name = args.train_list or self.config['data']["lists"]["train"]
        self.test_list_name = args.test_list or self.config['data']["lists"]["test"]


        self.architecture_key = self.config["trainer"]["model"]['name']
        self.architecture_args_dict = {k:v for k,v in self.config["trainer"]["model"].items() if k != 'name'}
        
        self.criterion_key = self.config["trainer"]["criterion"]['name']
        self.criterion_args_dict = {k:v for k,v in self.config["trainer"]["criterion"].items() if k != 'name'}
        
        self.optimizer_key = self.config["trainer"]["optimizer"]['name']
        self.optimizer_args_dict = {k:v for k,v in self.config["trainer"]["optimizer"].items() if k != 'name'}
        
        self.scheduler_key = self.config["trainer"]["scheduler"]['name']
        self.scheduler_args_dict = {k:v for k,v in self.config["trainer"]["scheduler"].items() if k != 'name'}
        


        self.num_channels = args.num_channels or self.config['trainer']['model']['in_channels']

        self.device = args.device or self.config['trainer']['device']
        self.description = args.description or self.config['trainer']['description']

        self.seed = self.config['trainer']['seed'] or 0

        self.weights_path = args.model_weights or self.config['trainer']['weights']
        
        self.DEBUG = args.DEBUG or self.config['DEBUG']


        ## UPDATE CONFIG FILE WITH PARSED ARGUMENTS
        from flatten_dict import flatten, unflatten
        tmp_dict = flatten(self.config)
        tmp_dict[("trainer", "epochs")] = self.EPOCHS
        tmp_dict[("trainer", "batch_size")] = self.BATCHSIZE
        tmp_dict[('data', "lists", "train")] = self.train_list_name
        tmp_dict[('data', "lists", "test")] = self.test_list_name
        tmp_dict[('trainer', 'model', 'in_channels')] = self.num_channels
        tmp_dict[('trainer', 'device')] = self.device
        tmp_dict[('trainer', 'description')] = self.description
        tmp_dict[('trainer', 'seed')] = self.seed
        tmp_dict[('trainer', 'weights')] = self.weights_path
        tmp_dict[('DEBUG',)] = self.DEBUG
        self.config = unflatten(tmp_dict)

    def __repr__(self) -> str:
        return f"""
Net Options:

  Configuration file: {self.cfg_name}
  Seed: {self.seed}
  Weights: {self.weights_path}
  Description: {self.description}

  Trainer:
    User number: {self.USER_NO}
    Epochs: {self.EPOCHS}
    Batch Size: {self.BATCHSIZE}
    Architecture: {self.architecture_key}
    Criterion: {self.criterion_key}
    Optimizer: {self.optimizer_key}
    Scheduler: {self.scheduler_key}

  Data:
    Frame Integration Time (dT): {self.dT}  ms
    Learnable Video Duration (subclipT): {self.subclipT}  ms
    Stride: {self.stride}
    Data Type: {self.data_type}

  Lists:
    Train List Name: {self.train_list_name}
    Test List Name: {self.test_list_name}
"""

    
class FileManager:
    
    def __init__(self, netopts: NetConfig, date: str) -> None:
        self.date = date
        self.files, self.dirs, self.model_names = self.get_files(netopts=netopts)
        self.create_dirs(self.dirs)
    
    def create_dirs(self, dirs_dict: Dict) -> None:
        [os.makedirs(pth) for pth in dirs_dict.values() if not os.path.exists(pth)]
         
    def get_files(self, netopts: NetConfig) -> Union[Dict[str,str], Dict[str,str], Dict[str,str], str]:

        paths = get_multiple_paths(levels=(
                                            ("model", "logs"), 
                                            ("model", "weights"), 
                                            ("model", "tests"), 
                                            ("dry-run", "logs"), 
                                            ("dry-run", "weights"),
                                            ('lists')
                                            ))

        # DIRS
        self.dir_logs, self.dir_model, self.dir_tests, self.dir_dry_logs, self.dir_dry_model, self.dir_lists = paths
        
        # MODEL NAMES
        self.MODEL_NAME = f"model_{self.date}" # EXP -> experience number ; s -> stride ; c -> number of classes ; dur -> clip duration
        # self.MODEL_NAME = f"model_EXP{netopts.exp:02d}_{self.date}_s{netopts.stride}_c{netopts.USER_NO}_dur{netopts.subclipT}_{netopts.reppath}" # EXP -> experience number ; s -> stride ; c -> number of classes ; dur -> clip duration
        self.DRY_MODEL_NAME = "model_dry"

        # FILES
        self.model_options_log = os.path.join(self.dir_logs, f"{self.MODEL_NAME}.log")
        self.model_log = os.path.join(self.dir_logs, f"{self.MODEL_NAME}.txt")
        self.dry_model_log = os.path.join(self.dir_dry_logs, f"{self.DRY_MODEL_NAME}.txt")

        self.test_file = os.path.join(self.dir_tests, f"test_{self.MODEL_NAME}_{Path(netopts.test_list_name).stem}.json")

        self.train_list_file = os.path.join('' if Path(netopts.train_list_name).name != netopts.train_list_name else self.dir_lists, netopts.train_list_name) 
        self.test_list_file = os.path.join('' if Path(netopts.test_list_name).name != netopts.test_list_name else self.dir_lists, netopts.test_list_name) 
        

        model_names={
                'MODEL_NAME':self.MODEL_NAME,
                'DRY_MODEL_NAME':self.DRY_MODEL_NAME,
        }
        dirs = {
                'dir_logs':self.dir_logs,
                'dir_model':self.dir_model,
                'dir_tests':self.dir_tests,
                'dir_dry_logs':self.dir_dry_logs,
                'dir_dry_model':self.dir_dry_model,
        }
        files = { 
                'model_options_log':self.model_options_log,
                'model_log':self.model_log,
                'dry_model_log':self.dry_model_log,
                'test_file':self.test_file,
                'train_list_file':self.train_list_file,
                'test_list_file':self.test_list_file,
                }
        
        return files, dirs, model_names

    def get_file(self, key:str) -> str:
        return self.files[key]
    
    def get_dir(self, key:str) -> str:
        return self.dirs[key]
    
    def get_model_name(self, key:str) -> str:
        return self.model_names[key]
    
    def get_date(self) -> str:
        return self.date
    
    def __repr__(self) -> str:
        return f"""
Files and Directories:
  Model names:
{dict2str(self.model_names, level=1)}
  Directories:
{dict2str(self.dirs, level=1)}
  Files:
{dict2str(self.files, level=1)}
  """
    
    











class NetConfigV2:
    def __init__(self, file_list, args):
        self.cfg_name = '&'.join(deepcopy(file_list))
        self.config = Configurations.load_multiple_cfg(cfg_list=file_list)
        

        self.USER_NO = self.config["trainer"]["model"]['num_classes']
        self.EPOCHS = args.epochs or self.config["trainer"]["epochs"]
        self.BATCHSIZE = args.batch_size or self.config["trainer"]["batch_size"]

        self.dT = self.config["data"]['dataset']["dT"]  # Frame Time: 40 ms
        self.subclipT = self.config["data"]['dataset']["subclipT"]  # Learnable Video Duration: 500 ms
        self.stride = self.config["data"]['dataset']["stride"]

        self.data_type, self.reppath = self.config["data"]['dataset']["type"].split('_') if len(self.config["data"]['dataset']["type"].split('_')) > 1 else (self.config["data"]['dataset']["type"], '')
        
        self.train_list_name = args.train_list or self.config['data']["lists"]["train"]
        self.test_list_name = args.test_list or self.config['data']["lists"]["test"]


        def get_name_kwargs_from_trainer_cfg(module:str):
            name=self.config["trainer"][module].pop('name')
            kwargs={k:v for k,v in self.config["trainer"][module].items() }
            return name, kwargs


        self.architecture_key, self.architecture_args_dict = get_name_kwargs_from_trainer_cfg("model")
        self.criterion_key, self.criterion_args_dict = get_name_kwargs_from_trainer_cfg("criterion")
        self.optimizer_key, self.optimizer_args_dict = get_name_kwargs_from_trainer_cfg("optimizer")
        self.scheduler_key, self.scheduler_args_dict = get_name_kwargs_from_trainer_cfg("scheduler")


        self.num_channels = args.num_channels or self.config['trainer']['model']['in_channels']

        self.device = args.device or self.config['trainer']['device']
        self.description = args.description or self.config['trainer']['description']

        self.seed = self.config['trainer']['seed'] or 0

        self.weights_path = args.model_weights or self.config['trainer']['weights']
        
        self.DEBUG = args.DEBUG or self.config['DEBUG']


        ## UPDATE CONFIG FILE WITH PARSED ARGUMENTS
        from flatten_dict import flatten, unflatten
        tmp_dict = flatten(self.config)

        tmp_dict[("trainer", "epochs")] = self.EPOCHS
        tmp_dict[("trainer", "batch_size")] = self.BATCHSIZE
        tmp_dict[('data', "lists", "train")] = self.train_list_name
        tmp_dict[('data', "lists", "test")] = self.test_list_name
        tmp_dict[('trainer', 'model', 'in_channels')] = self.num_channels
        tmp_dict[('trainer', 'device')] = self.device
        tmp_dict[('trainer', 'description')] = self.description
        tmp_dict[('trainer', 'seed')] = self.seed
        tmp_dict[('trainer', 'weights')] = self.weights_path
        tmp_dict[('DEBUG',)] = self.DEBUG
        
        self.config = unflatten(tmp_dict)

    def __repr__(self) -> str:
        return f"""
Net Options:

  Configuration file: {self.cfg_name}
  Seed: {self.seed}
  Weights: {self.weights_path}
  Description: {self.description}

  Trainer:
    User number: {self.USER_NO}
    Epochs: {self.EPOCHS}
    Batch Size: {self.BATCHSIZE}
    Architecture: {self.architecture_key}
    Criterion: {self.criterion_key}
    Optimizer: {self.optimizer_key}
    Scheduler: {self.scheduler_key}

  Data:
    Frame Integration Time (dT): {self.dT}  ms
    Learnable Video Duration (subclipT): {self.subclipT}  ms
    Stride: {self.stride}
    Data Type: {self.data_type}

  Lists:
    Train List Name: {self.train_list_name}
    Test List Name: {self.test_list_name}
"""

    
# ENDFILE
