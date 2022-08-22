from typing import Any

from ..env import *
from ..tools.tools_basic import dict2str, update_nested_dict


class Configurations:
    """ Configuration files class"""

    @staticmethod
    def load_cfg(file:str, base_file:str='nebfir/config/base_cfg.yml') -> Dict:
        base_files = glob('configs/*.yml') + glob('nebfir/config/*.yml')
        cfg_files  = glob('configs/*.yml') + glob('data/out/model_logs/*.yml')

        assert base_file in base_files, f'Expected base file to be one of {base_files}. Got "{base_file}"'
        assert file in cfg_files, f'Expected file to be one of {cfg_files}. Got "{file}"'

        with open(base_file) as f: base_cfg = yaml.safe_load(f)
        with open(file) as f: cfg_dict = yaml.safe_load(f)
        
        if cfg_dict is None: return base_cfg
        
        return update_nested_dict(base_cfg, cfg_dict)

    @staticmethod
    def load_multiple_cfg(cfg_list: Union[str, List[str]]) -> Dict:
        """ Creates a dictionary with trainer and dataset paramenters from multile configuration files
        
        Warning! This method uses pop to get the first cfg file

        Args:
            cfg_list (Union[str, List[str]]): cfg files

        Returns:
            Dict: trainer and dataset paramenters dictionary
        """
        assert all([type(cfg) is str for cfg in cfg_list]), f'Expected all cfg files to be of type str. Got {[type(cfg) for cfg in cfg_list]}'
        assert isinstance(cfg_list, (str, list, tuple)), f'Expected configuration list to be of type (str |List | Tuple). Got type: {type(cfg_list)}'
        assert len(cfg_list) >= 1, f'Configuration list must contain at least 1 configuration file. Got length: {1 if isinstance(cfg_list, str) else len(cfg_list)}'
        
        cfg0 = cfg_list if isinstance(cfg_list, str) else cfg_list.pop(0)
        old_cfg_dict = Configurations.load_cfg(cfg0)

        if len(cfg_list) < 1 or isinstance(cfg_list, str): return old_cfg_dict
        
        for cfg_file in cfg_list:
            with open(cfg_file) as f: old_cfg_dict = update_nested_dict(old_cfg_dict, yaml.safe_load(f))

        return old_cfg_dict

    @staticmethod
    def get_config(filename:str) -> Dict:
        assert Path(filename).is_file(), f"{filename} is not a configuration file!"
        
        with open(str(filename), 'r') as yfile:
            config = yaml.safe_load(yfile)
        return config

    @staticmethod
    def get_all() -> Dict[str, Union[Dict[str, str], str, int]]:
        cwd = Path.cwd()
        config_paths = cwd / 'configs' / '*.y*ml' # Match .yml or .yaml
        
        available_configurations = glob(str(config_paths))
        configs = {}
        for conf in available_configurations:
            with open(conf, 'r') as yfile:
                configs[Path(conf).name] = yaml.safe_load(yfile)
        return configs
    
    @staticmethod
    def print_config(config: Union[Dict[str, Any], str]):
        if isinstance(config, Dict):
            pass
        elif isinstance(config, str):
            config = Configurations.get_config(config)
        else:
            raise TypeError(f'Wrong type for {config}. Type: {type(config)}')
        print(dict2str( config, indentation=2) ) 

    @staticmethod
    def print_all(configs: Dict = None):
        if configs is None:
            configs = Configurations.get_all()
        [print(key + '\n' + dict2str( config, indentation=2) + 2*'\n')  for key, config in configs.items()]

    @staticmethod
    def create_configuration(base_config_name:str = None, config_save_name:str=None):
        config_path = Path.cwd() / 'configs' / (Path(config_save_name).stem + '.yml')
        assert not config_path.is_file(), f"{config_path} is a configuration file!"
        

        base_config = Configurations.get_config(base_config_name or 'base-config.yml')

        def change_dict_values(dict_):
            for k, v in dict_.items():
                if isinstance(v, (str, int, float)):
                    dict_[k] = input(f'Input value for {k} (default:{v} -> {type(v)}): ') or v  # VALUE
                elif isinstance(v, dict):
                    dict_[k] = change_dict_values(v)  # VALUE
                else:
                    raise TypeError(f"Dictionary must contain values of either type: tuple[int or str], list[int or str], dict, int, str . Unsupported value: {v}")
            return dict_

        new_config = change_dict_values(base_config)

        if config_save_name is not None:
            Configurations.save_config(new_config, filename=config_save_name)
        

        return new_config

    @staticmethod
    def save_config(config, filename:str=None) -> Dict:       
        config_path = Path.cwd() / 'configs' / (Path(filename).stem + '.yml')
        assert not config_path.is_file(), f"{config_path} is a configuration file!"
        
        with open(str(config_path), 'w') as yfile:
            config = yaml.safe_dump(config, yfile, sort_keys=False)



     