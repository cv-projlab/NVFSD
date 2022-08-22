import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose

from ..dataloader.dataloader import EventsDatasetV2
from ..dataloader.transforms import affine, normalize, rnd_erase
from ..env import *
from ..metrics.metrics_new import AccuracyMetrics, LossMetrics
from ..model.net_builder import NetV2
from ..tools.tools_basic import GroupAsDict
from .files import Files
from .trainer_params import TrainerParams
from .weights import Weights


class Trainer:
    def __init__(self, args):
        self.date = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
        
        self.cfg = args.configuration
        
        trainer_params = TrainerParams(args)
        self.update_attributes_from(trainer_params)

        self.weights_manager = Weights()
        self.files_manager = Files(date=self.date, trainer_params=trainer_params)
        
        self.net = NetV2(trainer_params)

        self.create_dataloaders()
        self.create_metrics(loss_metrics=LossMetrics, acc_metrics=AccuracyMetrics)


        self.curr_epoch = self.weights_manager.load_checkpoint(self.weights_path, self.net.model, self.net.optimizer, self.net.scheduler)



    def update_attributes_from(self, clss):
        """ Updates class attributes

        IMPORTANT NOTE: only instance attributes are used for the class update. 

        Args:
            clss (class): Initialized class
        """
        clss_dict = dict(filter(lambda x: x if not '__' in x[0] else None, clss.__dict__.items()))
        [setattr(self, k, v) for k,v in clss_dict.items()]

    def create_dataloaders(self):
        available_tfs = [rnd_erase, affine]
        
        Ttforms = Compose([*[tf() for tf in available_tfs if tf.__name__ in self.transforms_list], normalize()])
        TTtforms = Compose([normalize()])

        dataset_kwargs = {
            'csv_file': 'data/inp/lists/SynFED_df_all_events_aets40.csv', 
            'userno':self.userno,
            'fakeno':self.fakeno,
            'recno':self.recno,
            'train_split_frac': .7, 
            'authentics':self.authentics, 
            'impostors':self.impostors, 
            'rand_state':self.seed, 
        }

        self.Tdataset = EventsDatasetV2(split = 'train', transform=Ttforms, **dataset_kwargs) # Train Dataset
        self.TTdataset = EventsDatasetV2(split = 'test', transform=TTtforms, **dataset_kwargs) # Test Dataset
        
        dataloader_kwargs = {
            'batch_size':self.batch_size, 
            'num_workers':self.workers,
            'pin_memory':self.pin_memory,
            'shuffle':True
        }
        self.Tdataloader = DataLoader(self.Tdataset, drop_last=True, **dataloader_kwargs) # Train Dataloader
        self.TTdataloader = DataLoader(self.TTdataset, drop_last=False, **dataloader_kwargs) # Test Dataloader

    def create_metrics(self, loss_metrics, acc_metrics):
        self.train_loss_metrics = loss_metrics(criterion=self.net.criterion)
        self.test_loss_metrics = loss_metrics(criterion=self.net.criterion)
        self.train_acc_metrics = acc_metrics()
        self.test_acc_metrics = acc_metrics()

        metrics_dict={
            'train_loss_metrics':self.train_loss_metrics,
            'test_loss_metrics':self.test_loss_metrics,
            'train_acc_metrics':self.train_acc_metrics,
            'test_acc_metrics':self.test_acc_metrics,
            }

        self.metrics_group = GroupAsDict(metrics_dict)


    def train(self):
        print("TRAIN!")

        bar = tqdm(range(self.curr_epoch, self.epochs))  
        for curr_epoch in bar:
            self.curr_epoch = curr_epoch

            self.metrics_group.apply('reset_metrics')

            # Train and Validate
            self.train_epoch()
            self.validate_epoch()

            self.net.scheduler.step(self.test_acc_metrics.mean_value)

            # Save iterations
            self.files_manager.save_trainer_iterations(**self.metrics_group.group)
            # self.files_manager.save_trainer_iterations(self.train_acc_metrics, self.train_loss_metrics, self.test_acc_metrics, self.test_loss_metrics)

            # Save model weights
            self.weights_manager.save_checkpoint(  
                curr_loss=self.test_loss_metrics.mean_value, 
                curr_acc=self.test_acc_metrics.mean_value, 
                model=self.net.model,
                optimizer=self.net.optimizer,
                scheduler=self.net.scheduler,
                epoch=self.curr_epoch, 
                run_dir=self.files_manager.run_dir, 
                model_name=self.files_manager.MODEL_NAME
                )
            


        # self.tensorboard_writer.add_hparams(
        #     {
        #         'net':self.architecture_key, 
        #         'criterion':self.criterion_key, 
        #         'optimizer':self.optimizer_key, 
        #         'scheduler':self.scheduler_key, 
        #         "lr": self.optimizer_args_dict['lr'], 
        #         "batchsize": self.batch_size, 
        #         'checkpoint':self.weights_path}, 
        #     {
        #         "accuracy": self.test_acc_metrics.mean_value, 
        #         "loss": self.test_loss_metrics.mean_value }, 
        #     run_name='hparams')
    
    
        [self.test(self.batch_size, progress=progress) for progress in glob(f"data/runs/{self.files_manager.MODEL_NAME}/{self.files_manager.MODEL_NAME}*.pth", recursive=True)]

    @torch.no_grad()
    def _validate(self, input_batch, labels):
        val_outputs = self.net.model(input_batch)
        
        self.test_acc_metrics.update(val_outputs, labels, convert_target2onehot=False)
        self.test_loss_metrics.update(val_outputs, labels, convert_target2onehot=False)


        pass







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

    def train(self):
        print("TRAIN!")

        MODEL_NAME = self.files_manager.MODEL_NAME 
        log_file = self.files_manager.model_iterations 
        run_dir = self.files_manager.run_dir 

        bar = tqdm(range(self.netopts.EPOCHS))  
        for epoch in bar:
            self.epoch = epoch

            self.train_acc_metrics.reset()
            self.train_loss_metrics.reset()
            self.test_acc_metrics.reset()
            self.test_loss_metrics.reset()

            # Train and Validate
            self.train_epoch()
            self.validate_epoch()

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
            self.weights_manager.save(  curr_loss=self.test_loss_metrics.mean_value, 
                                        curr_acc=self.test_acc_metrics.mean_value, 
                                        model=self.net.model, 
                                        run_dir=run_dir, 
                                        model_name=MODEL_NAME)
            


        self.tensorboard_writer.add_hparams({'net':self.netopts.architecture_key, 'criterion':self.netopts.criterion_key, 'optimizer':self.netopts.optimizer_key, 'scheduler':self.netopts.scheduler_key, "lr": self.netopts.config['trainer']['optimizer']['lr'], "batchsize": self.netopts.config['trainer']['batch_size'], 'pretrain':self.netopts.config['trainer']['weights']}, {"accuracy": self.test_acc_metrics.mean_value, "loss": self.test_loss_metrics.mean_value }, run_name='hparams')
    
    
        [self.test(self.netopts.BATCHSIZE, progress=progress) for progress in glob(f"data/runs/{MODEL_NAME}/{MODEL_NAME}*.pth", recursive=True)]


    def dry(self):
        print("DRY RUN!")

        MODEL_NAME = self.files_manager.DRY_MODEL_NAME
        run_dir = self.files_manager.dry_run_dir
        log_file = self.files_manager.dry_model_log

        self.epoch = 0

        self.train_acc_metrics.reset()
        self.train_loss_metrics.reset()
        self.test_acc_metrics.reset()
        self.test_loss_metrics.reset()

        # Train and Validate
        self.train_epoch()
        self.validate_epoch()

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
        self.weights_manager.save(curr_loss=self.test_loss_metrics.mean_value, curr_acc=self.test_acc_metrics.mean_value, model=self.net.model, run_dir=run_dir, model_name=MODEL_NAME)

        

# ENDFILE
