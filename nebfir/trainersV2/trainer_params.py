
from ..config.configurations import Configurations
from ..env import *
from ..tools.tools_basic import dict2str


class TrainerParams:
    def __init__(self, args) -> None:
        cfg = args.configuration

        cfg_params = Configurations.load_multiple_cfg(cfg_list=deepcopy(cfg))
        
        # MISC
        self.cfg = cfg
        self.seed = args.seed or cfg_params['trainer']['seed']


        # TRAINER
        self.userno = args.userno or cfg_params['trainer']['model']['num_classes']
        self.description = args.description or cfg_params['trainer']['description']
        
        self.train_list_name = args.train_list or cfg_params['data']["lists"]["train"]
        self.test_list_name = args.test_list or cfg_params['data']["lists"]["test"]


        # NET
        self.device = args.device or cfg_params['trainer']['device']
        self.data_parallel = args.data_parallel or cfg_params['trainer']['data_parallel']
        self.weights_path = args.model_weights or cfg_params['trainer']['weights']
        
        self.num_channels = args.num_channels or cfg_params['trainer']['model']['in_channels']
        
        self.architecture_key = cfg_params["trainer"]["model"]['name']
        self.architecture_args_dict = {k:v for k,v in cfg_params["trainer"]["model"].items() if k != 'name'}

        self.criterion_key = cfg_params["trainer"]["criterion"]['name']
        self.criterion_args_dict = {k:v for k,v in cfg_params["trainer"]["criterion"].items() if k != 'name'}

        self.optimizer_key = cfg_params["trainer"]["optimizer"]['name']
        self.optimizer_args_dict = {k:v for k,v in cfg_params["trainer"]["optimizer"].items() if k != 'name'}

        self.scheduler_key = cfg_params["trainer"]["scheduler"]['name']
        self.scheduler_args_dict = {k:v for k,v in cfg_params["trainer"]["scheduler"].items() if k != 'name'}


        # DATALOADER
        self.epochs = args.epochs or cfg_params['trainer']['epochs']
        self.batch_size = args.batch_size or cfg_params["trainer"]["batch_size"]
        
        self.transforms_list = cfg_params['trainer']['transforms']

        self.workers = cfg_params['trainer']['workers']
        self.pin_memory = cfg_params['trainer']['pin_memmory']


        # DATASET
        self.recno = cfg_params['data']['dataset']['recno']
        self.fakeno = cfg_params['data']['dataset']['fakeno']

        self.impostors = cfg_params['data']['dataset']['impostors']
        self.authentics = cfg_params['data']['dataset']['authentics']






        ## UPDATE CONFIG FILE WITH PARSED ARGUMENTS
        from flatten_dict import flatten, unflatten
        tmp_dict = flatten(cfg_params)

        tmp_dict[('trainer','seed')] = self.seed
        tmp_dict[('trainer','epochs')] = self.epochs
        tmp_dict[("trainer","batch_size")] = self.batch_size
        tmp_dict[('trainer','model','num_classes')] = self.userno
        tmp_dict[('data',"lists","train")] = self.train_list_name
        tmp_dict[('data',"lists","test")] = self.test_list_name
        tmp_dict[('trainer','description')] = self.description
        tmp_dict[('trainer','device')] = self.device
        tmp_dict[('trainer','data_parallel')] = self.data_parallel
        tmp_dict[('trainer','weights')] = self.weights_path
        tmp_dict[('trainer','model','in_channels')] = self.num_channels
        
        self.updated_cfg_params = unflatten(tmp_dict)

        

    def __repr__(self) -> str:
        return dict2str(self.__dict__, level=1, indentation=2)
        
# ENDFILE
