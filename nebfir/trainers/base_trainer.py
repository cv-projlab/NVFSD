from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

from ..dataloader import EventsDataset
from ..dataloader.transforms import affine, normalize, rnd_erase
from ..env import *
from ..imop import transformations as T  # pyx transformations
from ..metrics.metrics_new import Metrics
from ..model.net_builder import Net
from ..tools.tools_basic import tprint
from ..tools.tools_path import is_file, join_paths
from ..tools.tools_visualization import view_multi_frames_cv2
from ..trainers.files import Files
from .files_manager import FileManager, Logger, NetConfig
from .trainer_enums import metrics_enum


class BaseTrainer:
    date = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")

    def __init__(self, args, view_progress=True) -> None:
        self.config = args.configuration

        self.netopts = NetConfig(file_list=self.config, args=args)
        self.DEBUG = self.netopts.config['DEBUG']
        
        self._init_seed(seed=self.netopts.seed)

        self.files = Files(date=self.date, net_params=self.netopts)

        self.transforms_list = self.netopts.config['trainer']['transforms']
        self._init_dataloaders()

        self.net = Net(netopts=self.netopts)

        # print('USING DATAPARALLEL !')
        # self.net.model = torch.nn.DataParallel(self.net.model)#, device_ids=list(range(torch.cuda.device_count())))


        self.is_using_triplet = self.netopts.config['trainer']['criterion']['name'] == 'batch_all_triplet'
        # print('self.is_using_triplet', self.is_using_triplet)
        loss_metric = metrics_enum.triplet_loss_metric.value if self.is_using_triplet else metrics_enum.loss_metric.value
        
        self._init_metrics(loss_metrics=loss_metric, acc_metrics=metrics_enum.accuracy_metric.value)
        # self._init_metrics(loss_metrics=LossMetrics, acc_metrics=AccuracyMetrics)

        self._init_dataset_vars()

        # tqdm
        self.view_progress = view_progress

        # Load weights
        if self.netopts.config['trainer']['weights']:
            self.load_progress(self.netopts.config['trainer']['weights'])
            
            
            


        self.logger = Logger()

        # Log Options
        logger_name = 'opts_logger'
        self.opts_logger = self.logger.setup_logger(logger_name, self.files.model_args_log, console_log=True)
        self.opts_logger.info(f'Trainer - {self.__class__.__name__}')
        self.opts_logger.info(self.netopts) # Log Net Options
        
        
        self.logger.remove_console_logger_handler(logger_name)
                
        self.opts_logger.info(self.files) # Log Files
        self.opts_logger.info(f'\nTrain {self.Tdataset}') # Log Train Dataset
        self.opts_logger.info(f'\nTest {self.TTdataset}') # Log Test Dataset
        
        self.opts_logger.info(f'Seed: {self.seed}')
        
        
        
        self.logger.add_console_logger_handler(logger_name)
        
        self.opts_logger.info(self.net) # Log Built Net
        
        self.opts_logger.info(f'\nTransforms_list: {self.transforms_list}')
        

        
        
        with open(self.files.model_args, 'w') as model_opts:
            yaml.dump(self.netopts.config , model_opts, sort_keys=False)
        



        self.tensorboard_writer = SummaryWriter(self.files.run_dir)
        self.tensorboard_current_train_iteration=0
        self.tensorboard_current_test_iteration=0

    def _init_seed(self, seed=0):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)


    def _init_dataloaders(self):
        available_tfs = [rnd_erase, affine]
        
        Ttforms = Compose([*[tf() for tf in available_tfs if tf.__name__ in self.transforms_list], normalize()])
        TTtforms = Compose([normalize()])


        self.Tdataset = EventsDataset(csv_file=self.files.train_list_file, transform=Ttforms)
        self.TTdataset = EventsDataset(csv_file=self.files.test_list_file, transform=TTtforms)
        
        batchsize = self.netopts.BATCHSIZE
        workers = self.netopts.config['trainer']['workers']
        pin_memory = self.netopts.config['trainer']['pin_memmory']
        # Train Dataloader
        self.Tdataloader = DataLoader(self.Tdataset, 
                                        batch_size=batchsize, 
                                        shuffle=True, 
                                        num_workers=workers,
                                        pin_memory=pin_memory,
                                        drop_last=True,)

        # Validation Dataloader
        self.TTdataloader = DataLoader(self.TTdataset, 
                                        batch_size=batchsize, 
                                        shuffle=True, 
                                        num_workers=workers,
                                        pin_memory=pin_memory,
                                        drop_last=False,)



    
    def _init_metrics(self, loss_metrics: Metrics, acc_metrics: Metrics):
        self.train_loss_metrics = loss_metrics(criterion=self.net.criterion, DEBUG=self.DEBUG)
        self.test_loss_metrics = loss_metrics(criterion=self.net.criterion, DEBUG=self.DEBUG)

        self.train_acc_metrics = acc_metrics(DEBUG=self.DEBUG)
        self.test_acc_metrics = acc_metrics(DEBUG=self.DEBUG)

        self.val_loss_min = 999.99
        self.val_acc_max = 0.0
        self.val_loss_mean = 0.0
        self.val_acc_mean = 0.0

    def _init_dataset_vars(self):
        self.framesPerClip = int(np.floor(self.netopts.subclipT / self.netopts.dT))
        self.clips_shape = (-1, self.netopts.num_channels, self.framesPerClip, 224, 224)  # (-1, no_channels, H, W)
        self.labels_shape = (-1, self.netopts.USER_NO)  # (-1 no_user)

    def load_progress(self, model_path='NO_MODEL_WEIGHTS'):
        if not Path(model_path).is_file():
            tprint(f'File not found: {model_path}')
            return

        # self.net.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)

        model_sd=torch.load(model_path, map_location='cpu')

        # new_model_sd = OrderedDict()
        # for k,v in model_sd.items():
        #     new_model_sd[k.replace('module.', '')] = v
        # del model_sd
        # self.net.model.load_state_dict(new_model_sd, strict=True)
        
        if isinstance(self.net.model, nn.DataParallel):
            self.net.model.module.load_state_dict(model_sd, strict=True)
        else:
            self.net.model.load_state_dict(model_sd, strict=True)

        self.net.model.to(self.netopts.device)
        # self.net.model = torch.nn.DataParallel(self.net.model)
        tprint('Loaded model!')



    def get_progress_viewer(self, iter_, colour=None):
        return tqdm(iter_, bar_format="{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]", colour=colour, position=0)

    def set_viewer(self, key, bar=None):
        if bar is None:
            return

        if key == "desc_train":
            bar.set_description(f"Epoch {self.epoch: 2}, lr: {self.net.optimizer.param_groups[0]['lr']:.2e}")
        elif key == "desc_val":
            bar.set_description(f"Epoch {self.epoch: 2},     [VAL]   ")
        elif key == "postfix_train":
            bar.set_postfix(
                acc=f"{self.train_acc_metrics.iteration_value.item(): >6.2f}",
                acc_mean=f"{self.train_acc_metrics.mean_value: >6.2f}",
                loss=f"{self.train_loss_metrics.iteration_value.item(): >6.3f}",
            )
        elif key == "postfix_val":
            bar.set_postfix(
                val_acc=f"{self.test_acc_metrics.iteration_value.item(): >6.2f}",
                val_acc_mean=f"{self.test_acc_metrics.mean_value: >6.2f}",
                val_loss=f"{self.test_loss_metrics.iteration_value.item(): >6.3f}",
            )
        else:
            raise KeyError(f"The key: {key} , is not accepted")


    def _train_epoch(self, input_batch, labels):
        self.net.optimizer.zero_grad()

        if self.DEBUG:
            view_multi_frames_cv2(input_batch[0].cpu().numpy())

        # FORWARD PASS
        outputs = self.net.model(input_batch) if not self.is_using_triplet else self.net.model(input_batch, True)
        # outputs = self.net.model(input_batch, embeddings_flag=self.is_using_triplet)

        self.train_acc_metrics.update(outputs[1].view(*self.labels_shape) if self.is_using_triplet else outputs, labels, convert_target2onehot=False)
        loss = self.train_loss_metrics.update(outputs, labels, convert_target2onehot=False)

        # BACKWARD PASS
        loss.backward()
        self.net.optimizer.step()


    def train_epoch(self, dry=-1):
        self.net.model.train()

        epoch = self.get_progress_viewer(self.Tdataloader) if self.view_progress else self.Tdataloader
        self.set_viewer(key="desc_train", bar=epoch)

        self.val_loss_mean = 0
        self.val_acc_mean = 0

        for counter, data in enumerate(epoch):
            input_batch = data.get("clip").view(*self.clips_shape).to(self.net.device)
            labels = data.get("ohlabel").long().view(*self.labels_shape).to(self.net.device)
            
            self._train_epoch(input_batch, labels)

            self.tensorboard_writer.add_scalar('Train/Accuracy', self.train_acc_metrics.iteration_value, self.tensorboard_current_train_iteration)
            self.tensorboard_writer.add_scalar('Train/Loss', self.train_loss_metrics.iteration_value, self.tensorboard_current_train_iteration)
            self.tensorboard_current_train_iteration+=1
            self.set_viewer(key="postfix_train", bar=epoch)

            if counter == dry: break

    def _validate_epoch(self, input_batch, labels):
        val_outputs = self.net.model(input_batch) if not self.is_using_triplet else self.net.model(input_batch, True)
        # val_outputs = self.net.model(input_batch, embeddings_flag=self.is_using_triplet)
        
        self.test_acc_metrics.update(val_outputs[1].view(*self.labels_shape) if self.is_using_triplet else val_outputs, labels, convert_target2onehot=False)
        self.test_loss_metrics.update(val_outputs, labels, convert_target2onehot=False)



        return val_outputs

    def validate_epoch(self, dry=-1):
        self.net.model.eval()

        tepoch_val = self.get_progress_viewer(self.TTdataloader, colour="green") if self.view_progress else self.TTdataloader
        self.set_viewer("desc_val", tepoch_val)

        with torch.no_grad():
            for counter, data in enumerate(tepoch_val):
                val_input_batch = data.get("clip").view(*self.clips_shape).to(self.net.device)
                val_labels = data.get("ohlabel").long().view(*self.labels_shape).to(self.net.device)
                
                self._validate_epoch(val_input_batch, val_labels)
                
                self.tensorboard_writer.add_scalar('Test/Accuracy', self.test_acc_metrics.iteration_value, self.tensorboard_current_test_iteration)
                self.tensorboard_writer.add_scalar('Test/Loss', self.test_loss_metrics.iteration_value, self.tensorboard_current_test_iteration)
                self.tensorboard_current_test_iteration+=1
                self.set_viewer("postfix_val", tepoch_val)

                if counter == dry: break

    def train(self, dry_len=-1):
        if dry_len < 0:
            print("TRAIN!")
        else:
            print("DRY RUN!")

        log_file = self.files.model_iterations if dry_len < 0 else self.files.dry_model_log
        # log_file = self.fm.model_log if dry_len < 0 else self.fm.dry_model_log
        MODEL_NAME = self.files.MODEL_NAME if dry_len < 0 else self.files.DRY_MODEL_NAME
        model_dir = self.files.run_dir if dry_len < 0 else self.files.dry_run_dir

        bar = tqdm(range(self.netopts.EPOCHS))  
        for epoch in bar:
            self.epoch = epoch

            self.train_acc_metrics.reset()
            self.train_loss_metrics.reset()
            self.test_acc_metrics.reset()
            self.test_loss_metrics.reset()

            # Train and Validate
            self.train_epoch(dry=dry_len)
            self.validate_epoch(dry=dry_len)

            self.net.scheduler.step(self.test_acc_metrics.mean_value)


            train_data = np.array(
                [["train"] * len(self.train_acc_metrics.iteration_value_list), 
                 self.train_acc_metrics.iteration_value_list, 
                 self.train_loss_metrics.iteration_value_list,]
            ).transpose()
            test_data = np.array(
                [["val"] * len(self.test_acc_metrics.iteration_value_list), 
                 self.test_acc_metrics.iteration_value_list, 
                 self.test_loss_metrics.iteration_value_list,]
            ).transpose()
            pd.DataFrame(np.vstack([train_data, test_data]), columns=["part", "acc", "loss"]).to_csv(log_file, index=False, mode="a")
            
            ##### SAVE MODEL ######
            self.save(dir_=model_dir, model_name=MODEL_NAME)

            if dry_len > -1: break
          


        self.tensorboard_writer.add_hparams({'net':self.netopts.architecture_key, 'criterion':self.netopts.criterion_key, 'optimizer':self.netopts.optimizer_key, 'scheduler':self.netopts.scheduler_key, "lr": self.netopts.config['trainer']['optimizer']['lr'], "batchsize": self.netopts.config['trainer']['batch_size'], 'pretrain':self.netopts.config['trainer']['weights']}, {"accuracy": self.test_acc_metrics.mean_value, "loss": self.test_loss_metrics.mean_value }, run_name='hparams')
    
    
        if dry_len is not None: self.net.reset()

        [self.test(self.netopts.BATCHSIZE, progress=prog) for prog in glob(f"data/runs/{self.files.MODEL_NAME}/{self.files.MODEL_NAME}*.pth")]
        # [self.test(self.netopts.BATCHSIZE, progress=prog) for prog in glob(f"data/out/weights/*{self.date}*.pth")]

    def test(self, batch_size=None, progress=''):
        self.net.model.eval()  
        
        if batch_size is None: batch_size=1

        if Path(progress).is_file(): self.load_progress(model_path=progress); 
        else: raise FileNotFoundError(f'File < {progress} > is not a valid file!')

        tprint(f"Test file: {Path(self.files.test_file).name}")
        self.test_acc_metrics.reset()
        self.test_loss_metrics.reset()

        # Test
        tepoch_val = self.get_progress_viewer(self.TTdataloader, colour="green") if self.view_progress else self.TTdataloader

        predictions = {'prediction':[], 'label':[], 'recording': [], 'user': []}
        for data in tepoch_val:
            val_input_batch = data.get("clip").view(*self.clips_shape).to(self.net.device)
            val_labels = data.get("ohlabel").long().view(*self.labels_shape).to(self.net.device)

            with torch.no_grad():
                val_outputs = self._validate_epoch(val_input_batch, val_labels)

            self.set_viewer("postfix_val", tepoch_val)

            predicted_class = torch.argmax(val_outputs[1].squeeze() if self.is_using_triplet else val_outputs.squeeze(), 1)
            
            item = lambda x: x.item()
            predictions['prediction'].extend(list(map(item, predicted_class)))
            predictions['label'].extend(list(map(item, data.get("label"))))
            predictions['recording'].extend(list(map(item, data.get("recording"))))
            predictions['user'].extend(list(map(item, data.get("user"))))
            

        test_file = self.files.test_file

        with open(join_paths(Path(test_file).parent, Path(test_file).stem + "_PREDICTIONS.json"), 'w') as jpred:
            json.dump(predictions, jpred) # indent=4

        with open(test_file, "w") as jfile:
            json.dump(
                {
                    "date": datetime.today().strftime("%d %B %Y, %Hh:%Mm:%Ss"),
                    "cfg": self.netopts.cfg_name,
                    "accuracy": f"{self.test_acc_metrics.mean_value}",
                    "model": Path(progress).stem,
                    # "model": self.files.MODEL_NAME,
                    "userno": self.netopts.USER_NO,
                    "train_list": self.netopts.train_list_name,
                    "test_list": self.netopts.test_list_name,
                    "description": self.netopts.description,
                },
                jfile,
                indent=4,
            )

    def save(self, dir_=None, model_name=None):
        if dir_ is None:
            dir_ = self.files.run_dir
        if model_name is None:
            model_name = self.files.MODEL_NAME

        val_loss_mean = self.test_loss_metrics.mean_value
        val_acc_mean = self.test_acc_metrics.mean_value

        flag = False
        if val_loss_mean < self.val_loss_min and not val_acc_mean > self.val_acc_max:
            self.val_loss_min = val_loss_mean
            flag = True
            name = f"{model_name}_minvalloss_ep"
            file_names_list = [name + "*"]

        elif val_acc_mean > self.val_acc_max and not val_loss_mean < self.val_loss_min:
            self.val_acc_max = val_acc_mean
            flag = True
            name = f"{model_name}_maxvalacc_ep"
            file_names_list = [name + "*"]

        elif val_loss_mean < self.val_loss_min and val_acc_mean > self.val_acc_max:
            self.val_loss_min = val_loss_mean
            self.val_acc_max = val_acc_mean
            flag = True
            name = f"{model_name}_bestvalaccloss_ep"
            file_names_list = [f"{model_name}_minvalloss_ep*", f"{model_name}_maxvalacc_ep*", name + "*"]

        if flag:
            self.save_ep_model(dir_=dir_, file_names_list=file_names_list, save_name=name)

        # torch.save(self.net.model.state_dict(), os.path.join(dir_, f"{model_name}.pth"))

        if isinstance(self.net.model, nn.DataParallel):
            torch.save(self.net.model.module.state_dict(), os.path.join(dir_, f"{model_name}.pth"))
        else:
            torch.save(self.net.model.state_dict(), os.path.join(dir_, f"{model_name}.pth"))


    def save_ep_model(self, dir_, file_names_list, save_name):
        for file_ in os.listdir(dir_):
            for filename in file_names_list:
                if fnmatch.fnmatch(file_, filename):
                    os.remove(os.path.join(dir_, file_))

        # torch.save(self.net.model.state_dict(), os.path.join(dir_, f"{save_name}{self.epoch}.pth"))

        if isinstance(self.net.model, nn.DataParallel):
            torch.save(self.net.model.module.state_dict(), os.path.join(dir_, f"{save_name}{self.epoch}.pth"))
        else:
            torch.save(self.net.model.state_dict(), os.path.join(dir_, f"{save_name}{self.epoch}.pth"))


# ENDFILE
