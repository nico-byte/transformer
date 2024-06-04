import os
import sys
import math
from lion_pytorch import Lion
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from nltk.translate.meteor_score import meteor_score
from src.transformer import Seq2SeqTransformer
from utils.config import TrainerConfig, SharedConfig
from time import perf_counter
from utils.logger import get_logger


class EarlyStopper:
    def __init__(self, warmup: int=5, patience: int=1, min_delta: int=0):
        self.warmup = warmup
        self.patience: int = patience
        self.min_delta: int = min_delta
        self.counter: int = 0
        self.min_validation_loss: float = float('inf')
        self.logger = get_logger('EarlyStopper')

    def early_stop(self, epoch, validation_loss):
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
        
        step_size = len(list(train_dataloader))
        
        CYCLE_STEPSIZE = (step_size / (trainer_config.tgt_batch_size / trainer_config.batch_size) * trainer_config.num_epochs) // 6
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=shared_config.special_symbols.index('<pad>'))
        self.optim = torch.optim.AdamW(self.model.parameters(), 
                                       lr=trainer_config.learning_rate, 
                                       amsgrad=True)
        
        self.scheduler = CyclicLR(self.optim, 
                                  base_lr=2e-6, 
                                  max_lr=trainer_config.learning_rate, 
                                  mode='triangular2', 
                                  step_size_up=CYCLE_STEPSIZE/2, 
                                  step_size_down=CYCLE_STEPSIZE/2, 
                                  cycle_momentum=False)
        """
        self.optim = Lion(model.parameters(), lr=trainer_config.learning_rate/3, betas=(0.95, 0.98), weight_decay=1e-2)
        self.scheduler = OneCycleLR(self.optim,
                                    max_lr=trainer_config.learning_rate, 
                                    total_steps=step_size*self.num_epochs, 
                                    anneal_strategy="cos", 
                                    cycle_momentum=False, 
                                    pct_start=0.3
                                    )                
        """
        self.device = device
        self.grad_accum: bool = trainer_config.tgt_batch_size > trainer_config.batch_size

        if self.grad_accum:
            self.accumulation_steps = trainer_config.tgt_batch_size // self.dataloaders[0].batch_size \
                if trainer_config.tgt_batch_size > self.dataloaders[0].batch_size else 1
                
    @classmethod
    def continue_training(cls, *args, **kwargs):
        return NotImplementedError
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return NotImplementedError

    def _train_epoch(self) -> float:
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

                all_preds = [[token for token in self.tokenizer.decode(pred_tokens) \
                    if token not in ["<bos>", "<eos>", "<pad>"]] for pred_tokens in predictions]
                all_targets = [[token for token in self.tokenizer.decode(tgt_tokens) \
                    if token not in ["<bos>", "<eos>", "<pad>"]] for tgt_tokens in targets]

                meteor = sum([meteor_score([all_targets[i]], preds) for i, preds in enumerate(all_preds) \
                    if len(preds) != 0]) / len(all_targets)
                avg_meteor += meteor

        return avg_meteor / len(list(self.dataloaders[-1]))

    def train(self):
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
        self._save_model_infer(name)
        self._save_model_train(name)
    
    def _save_model_infer(self, name=""):
        model_filepath = f'./models/{self.run_id}/{name}checkpoint_scripted.pt'
        
        model_scripted = torch.jit.script(self.model)
        model_scripted.save(model_filepath)
        self.logger.info(f'Saved model checkpoint to {model_filepath}')
        
    def _save_model_train(self, name=""):
        model_filepath = f'./models/{self.run_id}/{name}checkpoint.pt'
        
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            }, model_filepath)
        
        self.logger.info(f'Saved model checkpoint to {model_filepath}')
        
    def load_model(self):
        filepath = f'./models/{self.run_id}/checkpoint.pt'
        self.model = torch.jit.script(filepath)
        self.logger.info(f'Model checkpoint have been loaded from {filepath}')
