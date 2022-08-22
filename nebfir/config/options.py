import argparse
from glob import glob


def get_parser():
    parser = argparse.ArgumentParser(description="Experiments", add_help=False)
    parser.add_argument('-c', "--cfg", "--conf", "--config", "--configuration", type=str, required=True, choices=glob('configs/*.yml'), dest='configuration', help="Configuration file")

    return parser


def get_runner_options():
    parent_parser = get_parser()
    parser = argparse.ArgumentParser(description="Train Experiments", parents=[parent_parser])
    parser.add_argument("-f", "--function", type=str, required=True, help="Name of the function to run")

    args = parser.parse_args()
    return args


def get_options():
    parser = argparse.ArgumentParser(description="Trainer arguments")
    parser.add_argument('-c', "--cfg", "--conf", "--config", "--configuration", nargs='+', type=str, choices=glob('configs/*.yml')+glob('data/out/model_logs/*.yml'), dest='configuration', required=True, action='store', help="Configuration file")

    parser.add_argument("-w", "-p", "--weights", "--path", type=str, default=None, dest='model_weights', help="Model path")

    parser.add_argument("--tl", "--train_list", type=str, default=None, choices=glob('data/inp/lists/*'), dest='train_list', help="Choose a train list")
    parser.add_argument("--il", "--inference_list", "--test_list", type=str, default=None, choices=glob('data/inp/lists/*'), dest='test_list', help="Choose a test list")

    parser.add_argument("-d", "--dry", action='store_true', help="Dry run")
    parser.add_argument("-t", "--train", action='store_true', help="Train")
    parser.add_argument("-i", '--inference', "--test", action='store_true', dest='test', help="Test")

    parser.add_argument("--description", type=str, default="", help="Optional brief description about the training")
    parser.add_argument("--seed", type=str, default=None, help="Optional brief description about the training")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model")  # 'cuda:0'
    parser.add_argument("--data_parallel", type=bool, default=None, help="Use pytorch data_parallel")  # 'cuda:0'

    parser.add_argument('-u', "--users", "--userno", '--class_num', '--classes', type=int, default=None, dest='userno', help="Number of classes") 
    parser.add_argument('-ch', "--channels", type=int, choices=[1,3], default=1, dest='num_channels', help="Number of channels") 
    parser.add_argument('-b', "--batch_size", type=int, default=None, help="Batch Size") 
    parser.add_argument('-e', "--epochs", type=int, default=None, help="Train epochs") 
    
    parser.add_argument('--notif', '--email_error', '--error_notification', '--ERROR_NOTIFICATION', action='store_true', dest='error_notif', help="Email error notification")
    parser.add_argument('--debug', '--DEBUG', action='store_true', dest='DEBUG', help="Debug mode")

    args = parser.parse_args()
    return args


def get_jupyter_args(args_str='--cfg configs/config-0.yml'):
    parser = argparse.ArgumentParser(description="Trainer arguments")
    parser.add_argument('-c', "--cfg", "--conf", "--config", "--configuration", nargs='+', type=str, choices=glob('configs/*.yml')+glob('data/out/model_logs/*.yml'), dest='configuration', required=True, action='store', help="Configuration file")

    parser.add_argument("-w", "-p", "--weights", "--path", type=str, default=None, dest='model_weights', help="Model path")

    parser.add_argument("--tl", "--train_list", type=str, default=None, choices=glob('data/inp/lists/*'), dest='train_list', help="Choose a train list")
    parser.add_argument("--il", "--inference_list", "--test_list", type=str, default=None, choices=glob('data/inp/lists/*'), dest='test_list', help="Choose a test list")

    parser.add_argument("-d", "--dry", action='store_true', help="Dry run")
    parser.add_argument("-t", "--train", action='store_true', help="Train")
    parser.add_argument("-i", '--inference', "--test", action='store_true', dest='test', help="Test")

    parser.add_argument("--description", type=str, default="", help="Optional brief description about the training")
    parser.add_argument("--seed", type=str, default=None, help="Optional brief description about the training")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model")  # 'cuda:0'
    parser.add_argument("--data_parallel", type=bool, default=None, help="Use pytorch data_parallel")  # 'cuda:0'

    parser.add_argument('-u', "--users", "--userno", '--class_num', '--classes', type=int, default=None, dest='userno', help="Number of classes") 
    parser.add_argument('-ch', "--channels", type=int, choices=[1,3], default=1, dest='num_channels', help="Number of channels") 
    parser.add_argument('-b', "--batch_size", type=int, default=None, help="Batch Size") 
    parser.add_argument('-e', "--epochs", type=int, default=None, help="Train epochs") 
    
    parser.add_argument('--notif', '--email_error', '--error_notification', '--ERROR_NOTIFICATION', action='store_true', dest='error_notif', help="Email error notification")
    parser.add_argument('--debug', '--DEBUG', action='store_true', dest='DEBUG', help="Debug mode")

    args = parser.parse_args(args_str.split())
    return args

def get_test_options():
    parent_parser = get_parser()
    parser = argparse.ArgumentParser(description="Tets Experiments", parents=[parent_parser])
    parser.add_argument("-p", "--path", type=str, default=None, help="Model path")
    # parser.add_argument(
    #     "-s", "--static", type=bool, nargs="?", const=True, default=False, help="Use Dynamic or Static facial information (ONLY FOR TESTING)"
    # )
    # parser.add_argument(
    #     "-i",
    #     "--inverted",
    #     type=bool,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="Use Normal or Inverted (180°) facial information (ONLY FOR TESTING)",
    # )
    parser.add_argument("-l", "--test_list", type=str, default=None, choices=glob('data/inp/lists/*'), help="Choose a test list")
    parser.add_argument("-d", "--dry", action='store_true', help="Dry run")
    parser.add_argument("-t", "--train", action='store_true', help="Train")
    parser.add_argument("-i", '--inference', "--test", action='store_true', dest='test', help="Test")
    parser.add_argument("--description", type=str, default="", help="Optional brief description about the training")
    parser.add_argument("--device", type=str, default=None, help="Device to run the model")  # 'cuda:0'
    parser.add_argument('-ch', "--channels", type=int, choices=[1,3], default=1, dest='num_channels', help="Number of channels") 
    parser.add_argument('-b', "--batch_size", type=int, default=None, help="Batch Size") 
    
    parser.add_argument('-a', "--arch", "--architecture", type=str, default='i3d', choices={'i3d', 'tsf'}, dest='architecture', help="Model architecture") 

    args = parser.parse_args()
    return args


def get_activations_options():
    parser = argparse.ArgumentParser(description="Activations options")
    parser.add_argument("-e", "--experiment", type=int, default=3, help="Experiment number")

    parser.add_argument(
        "-d", "--date", type=str, defaults="2022-01-13_16h21m13s", help='Test date and time in format "[YYYY]-[MM]-[dd]_[HH]h[mm]m[ss]s"'
    )
    parser.add_argument(
        "-s", "--static", type=bool, nargs="?", const=True, default=False, help="Use Dynamic or Static facial information (ONLY FOR TESTING)"
    )
    parser.add_argument(
        "-i",
        "--inverted",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Use Normal or Inverted (180°) facial information (ONLY FOR TESTING)",
    )

    args = parser.parse_args()
    return args


def get_dataset_creation_options():
    parser = argparse.ArgumentParser(description="Datasets creation options")
    parser.add_argument(
        "-e",
        "--experiment",
        type=int,
        required=True,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 999],
        help="Experiment number",
    )
    parser.add_argument("-d", "--new_dataset", type=bool, nargs="?", const=True, default=False, help="Create new dataset flag")
    parser.add_argument("-l", "--new_list", type=bool, nargs="?", const=True, default=False, help="Create new list flag")
    parser.add_argument("-hl", "--help_lists", type=bool, nargs="?", const=True, default=False, help="Show how to use lists for the hdf dataset")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = get_test_options()
    args = get_dataset_creation_options()
    # args = get_options()
    print(args)
    print(type(args))
