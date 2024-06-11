import sys

import torch
import torch.nn as nn
from nltk.translate.meteor_score import meteor_score
from src.transformer import Seq2SeqTransformer
from utils.config import TrainerConfig, SharedConfig
from time import perf_counter
from utils.logger import get_logger

class InverseSquareRootLRScheduler:
    """
    Implements a learning rate scheduler with inverse square root decay.
    """

    def __init__(self, optimizer, init_lr, max_lr, n_warmup_steps):
        """
        Initialize the scheduler.

        Args:
            optimizer: The optimizer to adjust the learning rate for.
            init_lr (float): The initial learning rate.
            max_lr (float): The maximum learning rate.
            n_warmup_steps (int): The number of warmup steps.
        """

        self.optimizer = optimizer
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.n_warmup_steps = n_warmup_steps
        self.lr_step = (max_lr - init_lr) / n_warmup_steps
        self.decay_factor = max_lr * n_warmup_steps**0.5
        self.n_steps = 0

    def step(self):
        """
        Update the learning rate for the optimizer.
        """

        self.n_steps += 1
                
        if self.n_steps < self.n_warmup_steps:
            self.lr = self.init_lr + self.n_steps * self.lr_step
        else:
            self.lr = self.decay_factor * self.n_steps**-0.5
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_lr(self):
        """
        Get the current learning rate.
        """

        return self.optimizer.param_groups[0]['lr']


class EarlyStopper:
    """
    Implements early stopping to prevent overfitting during training.
    """

    def __init__(self, warmup: int=5, patience: int=1, min_delta: int=0):
        """
        Initialize the early stopper.

        Args:
            warmup: The number of warmup epochs. Defaults to 5.
            patience: The number of epochs to wait before stopping. Defaults to 1.
            min_delta: The minimum change in validation loss to qualify as an improvement. Defaults to 0.
        """

        self.warmup = warmup
        self.patience: int = patience
        self.min_delta: int = min_delta
        self.counter: int = 0
        self.min_validation_loss: float = float('inf')
        self.logger = get_logger('EarlyStopper')

    def early_stop(self, epoch, validation_loss):
        """
        Check if early stopping criterion is met.

        Args:
            epoch (int): The current epoch.
            validation_loss (float): The validation loss.

        Returns:
            bool: True if early stopping criterion is met, False otherwise.
        """

        if epoch < self.warmup:
            return False
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            self.logger.info(f'{self.counter} epochs without improvement. {self.patience - self.counter} epochs left unless model improves.')
            if self.counter >= self.patience:
                return True
        return False


class Trainer():
    """
    Class for training a sequence-to-sequence transformer model.
    """

    def __init__(self,
                 model: Seq2SeqTransformer,
                 translator,
                 train_dataloader,
                 test_dataloader,
                 val_dataloader,
                 tokenizer,
                 early_stopper: EarlyStopper,
                 trainer_config: TrainerConfig,
                 shared_config: SharedConfig,
                 run_id: str,
                 device):
        """
        Initialize the trainer.

        Args:
            model (Seq2SeqTransformer): The sequence-to-sequence transformer model.
            translator: The translator object.
            train_dataloader: The training dataloader.
            test_dataloader: The test dataloader.
            val_dataloader: The validation dataloader.
            tokenizer: The tokenizer.
            early_stopper (EarlyStopper): The early stopper object.
            trainer_config (TrainerConfig): The trainer configuration.
            shared_config (SharedConfig): The shared configuration.
            run_id (str): The ID for this training run.
            device: The device to run the training on.
        """        

        self.logger = get_logger('Trainer')
        
        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        self.model = model.to(device)
        self.translator = translator
        
        self.num_epochs = trainer_config.num_epochs
        
        self.current_epoch = 1
        
        self.dataloaders = [train_dataloader, test_dataloader, val_dataloader]
        self.tokenizer = tokenizer
        self.early_stopper = early_stopper
        
        self.run_id = run_id
        
                
        self.criterion = nn.CrossEntropyLoss(ignore_index=shared_config.special_symbols.index('<pad>'))
        self.optim = torch.optim.AdamW(self.model.parameters(), 
                                       lr=trainer_config.learning_rate, 
                                       amsgrad=True)
        
        self.scheduler = InverseSquareRootLRScheduler(optimizer=self.optim, 
                                                      init_lr=2e-6, 
                                                      max_lr=trainer_config.learning_rate, 
                                                      n_warmup_steps=trainer_config.warmup_steps)
        
        self.device = device
        self.grad_accum: bool = trainer_config.tgt_batch_size > trainer_config.batch_size

        if self.grad_accum:
            self.accumulation_steps = trainer_config.tgt_batch_size // self.dataloaders[0].batch_size \
                if trainer_config.tgt_batch_size > self.dataloaders[0].batch_size else 1
                
    @classmethod
    def continue_training(cls, *args, **kwargs):
        """
        Continue training from a checkpoint.

        Returns:
            NotImplementedError: Method not implemented.
        """

        return NotImplementedError
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """
        Load a pre-trained model for training.

        Returns:
            NotImplementedError: Method not implemented.
        """

        return NotImplementedError

    def _train_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns:
            float: The average training loss for the epoch.
        """

        self.model.train()
        losses = 0
        for batch_idx, (src, tgt) in enumerate(self.dataloaders[0]):
            tgt = tgt.type(torch.LongTensor)
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            
            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.translator.create_mask(src, tgt_input)
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
                tgt_out = tgt[1:, :]
                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            self.scaler.scale(loss).backward()
                        
            if self.grad_accum and (batch_idx + 1) % self.accumulation_steps == 0:
                for param in self.model.parameters():
                    param.grad /= self.accumulation_steps

                self.scaler.step(self.optim)
                self.scaler.update()
                self.scheduler.step()
                # Reset gradients, for the next accumulated batches
                for param in self.model.parameters():
                    param.grad = None
            else:
                self.scaler.step(self.optim)
                self.scaler.update()
                self.scheduler.step()
                for param in self.model.parameters():
                    param.grad = None   

            losses += loss.item()
        
        return losses / len(list(self.dataloaders[0]))


    def _test_epoch(self) -> float:
        """
        Test the model for one epoch.

        Returns:
            float: The average test loss for the epoch.
        """

        self.model.eval()
        losses = 0
        with torch.no_grad():
            for src, tgt in self.dataloaders[1]:
                tgt = tgt.type(torch.LongTensor)
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:-1, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.translator.create_mask(src, tgt_input)

                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
                    tgt_out = tgt[1:, :]
                    loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                
                losses += loss.item()

        return losses / len(list(self.dataloaders[1]))


    def evaluate(self) -> float:
        """
        Evaluate the model.

        Returns:
            float: The average meteor score for the evaluation dataset.
        """

        self.model.eval()
        avg_meteor = 0
        with torch.no_grad():
            for src, tgt in self.dataloaders[-1]:
                tgt = tgt.type(torch.LongTensor)
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:-1, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.translator.create_mask(src, tgt_input)

                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.use_amp):
                    logits = self.model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
                
                predictions = torch.argmax(logits, dim=-1)
                predictions = torch.tensor(predictions.T).cpu().numpy().tolist()
                targets = tgt_input.T.cpu().numpy().tolist()

                all_preds = [[token for token in self.tokenizer.encode(self.tokenizer.decode(pred)).tokens if token not in ["<bos>", "<eos>", "<pad>"]] for pred in predictions]
                all_targets = [[token for token in self.tokenizer.encode(self.tokenizer.decode(tgt)).tokens if token not in ["<bos>", "<eos>", "<pad>"]] for tgt in targets]
                
                meteor = sum([meteor_score([all_targets[i]], preds) for i, preds in enumerate(all_preds) \
                    if len(preds) != 0]) / len(all_targets)
                avg_meteor += meteor

        return avg_meteor / len(list(self.dataloaders[-1]))

    def train(self):
        """
        Train the model until convergence or early stopping.
        """

        try:
            for epoch in range(self.current_epoch, self.num_epochs+1):
                start_time = perf_counter()
                train_loss = self._train_epoch()
                self.logger.info(f'epoch {epoch} avg_training_loss: {round(train_loss, 3)} ({round(perf_counter()-start_time, 3)}s)')

                start_time = perf_counter()
                test_loss = self._test_epoch()
                self.logger.info(f'epoch {epoch} avg_test_loss: {round(test_loss, 3)} ({round(perf_counter()-start_time, 3)}s)')

                self.current_epoch += 1
                
                early_stop_true = self.early_stopper.early_stop(epoch, test_loss)
                counter = self.early_stopper.counter
                
                if counter == 1:
                    self._save_model("best_")
                
                if early_stop_true:
                    self._save_model("last_")
                    break
        except KeyboardInterrupt:
            self.logger.error('Training interrupted by user')
            self._save_model()
            sys.exit(0)
            
        self._save_model()
            
    def _save_model(self, name=""):
        """
        Save the model checkpoint.

        Args:
            name: The name of the model checkpoint. Defaults to "".
        """

        self._save_model_infer(name)
        self._save_model_train(name)
    
    def _save_model_infer(self, name=""):
        """
        Save the model checkpoint for inference.

        Args:
            name: The name of the model checkpoint. Defaults to "".
        """

        model_filepath = f'./models/{self.run_id}/{name}checkpoint_scripted.pt'
        
        model_scripted = torch.jit.script(self.model)
        model_scripted.save(model_filepath)
        self.logger.info(f'Saved model checkpoint to {model_filepath}')
        
    def _save_model_train(self, name=""):
        """
        Save the model checkpoint for further training.

        Args:
            name: The name of the model checkpoint. Defaults to "".
        """

        model_filepath = f'./models/{self.run_id}/{name}checkpoint.pt'
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),
            }, model_filepath)
        
        self.logger.info(f'Saved model checkpoint to {model_filepath}')
        
    def load_model(self):
        """
        Load the model from a checkpoint.
        """
    
        filepath = f'./models/{self.run_id}/checkpoint.pt'
        self.model = torch.jit.script(filepath)
        self.logger.info(f'Model checkpoint have been loaded from {filepath}')
