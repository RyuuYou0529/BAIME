import sys
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

from ..core.register import TRAINER_REGISTER
from ..arch import get_model
from ..dataset import get_dataset
from ..loss import get_loss
from ..optimizer import get_optimizer
from ..scheduler import get_scheduler

from ..utils.tensorboard import TBWriter
from ..utils.file import checkdir, Tee

@TRAINER_REGISTER.register('BaseTrainer')
class BaseTrainer(object):
    def __init__(self, args=None):
        super(BaseTrainer, self).__init__()

    def __init_attributes__(self, args=None, ctx=None):
        if not args:
            return
        self.args = args
        self.ctx = ctx

    def __tee__(self):
        stdout_log_path = os.path.join(self.args.out_log_path, f'worker_{self.ctx.rank}.out')
        stderr_log_path = os.path.join(self.args.out_log_path, f'worker_{self.ctx.rank}.err')
        sys.stdout = Tee(stdout_log_path, keep_stdout=True, keep_stderr=False)
        sys.stderr = Tee(stderr_log_path, keep_stdout=False, keep_stderr=True)

    def train_func(self, args, ctx):
        # === Initialize Attributes === #
        self.__init_attributes__(args, ctx)
        # === Prepare Tee for logging === #
        self.__tee__()

        # === Load Checkpoint if not Reset === #
        CKPT = self.load_checkpoint()
        # === Initialize Modules === #
        self.init_modules(CKPT)
        # === Let runtime post-process (e.g. DDP wrap, sampler injection) === #
        self.ctx.prepare_trainer(self)

        # === Training Loop === #
        start_epoch = self.start_epoch
        end_epoch = args.epochs
        for epoch in range(start_epoch, end_epoch):
            # === Train one Epoch === #
            avg_train_loss = self.train_one_epoch(epoch)

            # === Validate one Epoch === #
            avg_val_loss = self.val_one_epoch(epoch)

            # === Report Metrics and Checkpoints === #
            self.save_checkpoint(
                epoch, 
                metrics={'epoch': epoch+1, 'loss': avg_train_loss, 'val_loss': avg_val_loss}
            )
    
    def init_modules(self, CKPT=None):
        # === Prepare TensorBoard Writer === #
        if self.ctx.is_main:
            self.TBWriter = TBWriter(self.args.out_tb_path, self.args)

        # === Prepare Training DataLoader === #
        train_dataset = get_dataset(self.args, mode='train')
        self.TrainDataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size_per_worker,
            shuffle=self.args.shuffle,
            num_workers=self.args.workers,
        )

        # === Prepare Validation DataLoader === #
        if self.args.val_data_path:
            val_dataset = get_dataset(self.args, mode='val')
            self.ValDataloader = DataLoader(
                val_dataset,
                batch_size=self.args.batch_size_per_worker,
                shuffle=False,
                num_workers=self.args.workers,
            )

        # === Prepare Model === #
        model = get_model(self.args)
        if CKPT and 'model' in CKPT:
            model.load_state_dict(CKPT['model'])
        self.MODEL = model.to(self.ctx.device)

        # === Prepare Loss === #
        self.LOSS = get_loss(self.args)

        # === Prepare Optimizer === #
        self.Optimizer = get_optimizer(self.args, model)
        if CKPT and 'optimizer' in CKPT:
            self.Optimizer.load_state_dict(CKPT['optimizer'])
        
        # === Prepare Scheduler === #
        self.args.iters_per_epoch = len(self.TrainDataloader)
        self.Scheduler = get_scheduler(self.args)

        # === Set Start Epoch === #
        self.start_epoch = CKPT.get('epoch', 0) if CKPT else 0
        self.best_val_loss = CKPT.get('best_val_loss', None) if CKPT else None
    
    def cal_global_iter(self, epoch, batch_idx):
        num_workers = self.ctx.world_size
        worker_rank = self.ctx.rank
        global_iter = epoch*len(self.TrainDataloader)*num_workers + batch_idx*num_workers + worker_rank
        local_iter = epoch*len(self.TrainDataloader) + batch_idx
        return global_iter, local_iter

    def train_one_epoch(self, epoch):
        # === Set epoch for distributed sampler (if injected by runtime) === #
        if hasattr(self.TrainDataloader.sampler, 'set_epoch'):
            self.TrainDataloader.sampler.set_epoch(epoch)
        
        # === Training === #
        self.MODEL.train()
        total_loss = 0.0
        with tqdm(total=len(self.TrainDataloader), desc=f"TrainEpoch[{epoch+1}/{self.args.epochs}]", leave=True) as pbar:
            for batch_idx, (data, label) in enumerate(self.TrainDataloader):
                data, label = data.to(self.ctx.device), label.to(self.ctx.device)
                # === Calculate global iteration === #
                global_iter, local_iter = self.cal_global_iter(epoch, batch_idx)
                # === Update progress bar === #
                pbar.update(1)
                pbar.set_postfix({
                    "Batch": f"{batch_idx+1}/{len(self.TrainDataloader)}",
                })
                # === Set learning rate from scheduler === #
                current_lr = self.Scheduler.schedule[local_iter]
                for param_group in self.Optimizer.param_groups:
                    param_group["lr"] = current_lr
                # === Forward === #
                pred = self.MODEL(data)
                loss, loss_logger, image_logger = self.LOSS(pred, label)
                # === Zero Grad, Backward, Step === #
                self.Optimizer.zero_grad()
                loss.backward()
                self.Optimizer.step()

                total_loss += loss.item()

                # === Log Train Loss & Images & Learning Rate === #
                if self.ctx.is_main:
                    if loss_logger:
                        self.record_loss_logs(loss_logger, global_iter, prefix='Batch-Wise Train Loss')
                    if image_logger and (global_iter % 50 == 0):
                        self.record_image_logs(image_logger, global_iter, prefix='Batch-Wise Predictions')
                    self.record_learning_rate(current_lr, global_iter, prefix='Scheduler')
            avg_train_loss = total_loss / len(self.TrainDataloader)
            if self.ctx.is_main:
                self.record_loss_logs({'total': avg_train_loss}, epoch, prefix='Epoch-Wise Train Loss')
        return avg_train_loss
    
    def val_one_epoch(self, epoch):
        if not self.args.val_data_path:
            return None
        self.MODEL.eval()
        val_loss, num_total = 0.0, 0
        with torch.no_grad():
            for batch_idx, (data, label) in tqdm(enumerate(self.ValDataloader), desc=f'ValEpoch[{epoch+1}/{self.args.epochs}]'):
                data, label = data.to(self.ctx.device), label.to(self.ctx.device)
                pred = self.MODEL(data)
                loss, loss_logger= self.LOSS(pred, label)
                val_loss += loss.item()
                num_total += label.size(0)
        
        avg_val_loss = val_loss / len(self.ValDataloader)
        if self.ctx.is_main:
            self.record_loss_logs({'total': avg_val_loss}, epoch, prefix='Epoch-Wise Validation Loss')
        return avg_val_loss

    def record_loss_logs(self, loss_logger, it, prefix='Loss'):
        for key, value in loss_logger.items():
            self.TBWriter.add_scalars(f'{prefix}/{key}', value, global_step=it)
    
    def record_image_logs(self, image_logger, it, prefix='Images'):
        for key, value in image_logger.items():
            self.TBWriter.add_image(f'{prefix}/{key}', value, global_step=it)
    
    def record_learning_rate(self, lr, it, prefix='Scheduler'):
        self.TBWriter.add_scalars(f'{prefix}/LearningRate', lr, global_step=it)

    def load_checkpoint(self):
        if self.args.reset:
            print('Reset is True. Training from scratch.')
            return None
        ckpt_dir = self.args.out_ckpt_path
        ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, f'{self.args.model}_Epoch*.pt')))
        latest_ckpt_path = ckpt_files[-1] if ckpt_files else None
        if latest_ckpt_path:
            if latest_ckpt_path and os.path.isfile(latest_ckpt_path):
                print(f'Loading checkpoint from {latest_ckpt_path} ...')
                CKPT = torch.load(latest_ckpt_path, map_location='cpu', weights_only=False)
                return CKPT
            else:
                print('No checkpoint found. Training from scratch.')
                return None
        else:
            print('No checkpoint found. Training from scratch.')
            return None
            
    def save_checkpoint(self, epoch, metrics):
        if not self.ctx.is_main:
            return
        val_loss = metrics.get('val_loss')
        is_best = val_loss is not None and (self.best_val_loss is None or val_loss < self.best_val_loss)
        save_epoch = (epoch+1)%self.args.save_every == 0 or (epoch+1) == self.args.epochs
        if not is_best and not save_epoch:
            return

        if is_best:
            self.best_val_loss = val_loss
        ckpt_dir = self.args.out_ckpt_path
        checkdir(ckpt_dir, reset=False)
        model_state = {k.replace('module.', ''): v for k, v in self.MODEL.state_dict().items()}
        CKPT = {
            "epoch": epoch+1,
            "model": model_state,
            "optimizer": self.Optimizer.state_dict(),
            "args": self.args,
            "best_val_loss": self.best_val_loss
        }
        if is_best:
            torch.save(CKPT, os.path.join(ckpt_dir, f'{self.args.model}_BestValCKPT.pt'))
        if save_epoch:
            torch.save(CKPT, os.path.join(ckpt_dir, f'{self.args.model}_Epoch{str(epoch+1).zfill(4)}.pt'))
