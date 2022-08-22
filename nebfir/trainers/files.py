from ..env import *
from ..tools.tools_path import get_multiple_paths


class Files:
    def __init__(self, date: str, net_params) -> None:
        self.date = date
        self.net_params = net_params
        self.make_paths()

    @property
    def MODEL_NAME(self) -> str:
        return f"model_{self.date}"  
             
    def make_paths(self):
        
        # DIRS
        self.runs_dir, self.lists_dir = get_multiple_paths(levels=(('runs',), ('lists',)))
        self.run_dir = Path(self.runs_dir) / self.MODEL_NAME
        self.dry_run_dir = Path(self.runs_dir) / 'dry'

        full_path = lambda p: self.run_dir / p

        # FILES
        self.model_args = full_path(f'{self.MODEL_NAME}_args.yml')
        self.model_args_log = full_path(f'{self.MODEL_NAME}.log')
        self.model_iterations = full_path(f'{self.MODEL_NAME}.txt')
        self.model_test_predictions = full_path(f'{self.MODEL_NAME}_PREDICTIONS.json')
        self.model_test_results = full_path(f'{self.MODEL_NAME}.json')
        self.model_weights = full_path(f'{self.MODEL_NAME}.pth')



        # Make run dirs
        self.run_dir.mkdir(exist_ok=True, parents=True)
        self.dry_run_dir.mkdir(exist_ok=True, parents=True)

        self.DRY_MODEL_NAME="model"
        self.dry_model_log = os.path.join(self.dry_run_dir, self.DRY_MODEL_NAME+'.txt')


        self.train_list_file = os.path.join('' if Path(self.net_params.train_list_name).name != self.net_params.train_list_name else self.lists_dir, self.net_params.train_list_name) 
        self.test_list_file = os.path.join('' if Path(self.net_params.test_list_name).name != self.net_params.test_list_name else self.lists_dir, self.net_params.test_list_name) 

    @property
    def test_file(self) -> str:
        test_file = Path(self.run_dir) / self.model_test_results
        while test_file.is_file():
            test_file = Path(test_file).parent / (Path(test_file).stem + '_' + Path(test_file).suffix)
        return str(test_file)


    

    def __repr__(self) -> str:
        return f"""
Files and Directories:
  Directories:
    run dir: {self.run_dir}
    dryrun dir: {self.dry_run_dir}

  Files:
    args: {self.model_args}
    args_log: {self.model_args_log}
    iterations: {self.model_iterations}
    weights: {self.model_weights}
    test_predictions: {self.model_test_predictions}
    test_results: {self.model_test_results}
  """
    
    

