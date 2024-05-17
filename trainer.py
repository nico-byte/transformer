import os
import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR
from nltk.translate.meteor_score import meteor_score
from alive_progress import alive_bar
from transformer import Seq2SeqTransformer
from colorama import Fore, init
init(autoreset = True)


class EarlyStopper:
    def __init__(self, patience: int=1, min_delta: int=0):
        self.patience: int = patience
        self.min_delta: int = min_delta
        self.counter: int = 0
        self.min_validation_loss: float = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer():
    def __init__(self,
                 model: Seq2SeqTransformer,
                 early_stopper: EarlyStopper,
                 trainer_config,
                 shared_store, 
                 run_id: str,
                 resume: bool=False):
        if os.path.exists(f'./results/{run_id}') and not resume:
            print(f'{Fore.RED}Run ID already exists!')
            sys.exit(1)
        elif not resume:
            os.makedirs(f'./results/{run_id}')
        
        self.trainer_config = trainer_config
        self.shared_store = shared_store
        self.run_id = run_id
        
        STEPSIZE = (len(list(shared_store.dataloaders[0])) /
            (trainer_config.tgt_batch_size / trainer_config.batch_size) * trainer_config.num_epochs) // trainer_config.num_cycles
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=shared_store.special_symbols.index('<pad>'))
        self.optim = torch.optim.AdamW(self.model.parameters())
        self.scheduler = CyclicLR(self.optim, base_lr=2e-6, max_lr=trainer_config.learning_rate, mode='triangular2', 
                     step_size_up=STEPSIZE/2, step_size_down=STEPSIZE/2, cycle_momentum=False)
        
        self.early_stopper = early_stopper
        
        # resuming won't work because spacy tokenizer yields incosistent vocabs, so the dimensions are not alwys the same
        if resume:
            checkpoint = self._load_model_for_training()
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['optimizer_scheduler_dict'])
            self.epoch = checkpoint['epoch']
        else:
            self.epoch = 1
        
        self.dataloaders = shared_store.dataloaders
        
        self.device = trainer_config.device
        self.grad_accum: bool = trainer_config.tgt_batch_size > trainer_config.batch_size

        if self.grad_accum:
            self.accumulation_steps = trainer_config.tgt_batch_size // self.dataloaders[0].batch_size \
                if trainer_config.tgt_batch_size > self.dataloaders[0].batch_size else 1


    def _train_epoch(self) -> float:
        self.model.train()
        losses = 0
        for batch_idx, (src, tgt) in enumerate(self.dataloaders[0]):
            tgt = tgt.type(torch.LongTensor)
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            logits = self.model(src, tgt_input)

            self.optim.zero_grad()

            tgt_out = tgt[1:, :]

            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            if self.grad_accum and (batch_idx + 1) % self.accumulation_steps == 0:
                for param in self.model.parameters():
                    param.grad /= self.accumulation_steps

                self.optim.step()
                self.scheduler.step()
                # Reset gradients, for the next accumulated batches
                for param in self.model.parameters():
                    param.grad.zero_()
            else:
                self.optim.step()
                self.scheduler.step()

            losses += loss.item()

        return losses / len(list(self.dataloaders[0]))


    def _test_epoch(self) -> float:
        self.model.eval()
        losses = 0
        for src, tgt in self.dataloaders[1]:
            tgt = tgt.type(torch.LongTensor)
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            logits = self.model(src, tgt_input)

            tgt_out = tgt[1:, :]

            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(list(self.dataloaders[1]))


    def evaluate(self, tgt_language) -> float:
        self.model.eval()
        losses = 0
        avg_meteor = 0
        for src, tgt in self.dataloaders[-1]:
            tgt = tgt.type(torch.LongTensor)
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            logits = self.model(src, tgt_input)

            tgt_out = tgt[1:, :]

            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            predictions = torch.tensor(predictions.T).cpu().numpy().tolist()
            targets = tgt_input.T.cpu().numpy().tolist()

            all_preds = [[token for token in self.shared_store.vocab_transform[tgt_language].lookup_tokens(pred_tokens) \
                if token not in ["<bos>", "<eos>", "<pad>"]] for pred_tokens in predictions]
            all_targets = [[token for token in self.shared_store.vocab_transform[tgt_language].lookup_tokens(tgt_tokens) \
                if token not in ["<bos>", "<eos>", "<pad>"]] for tgt_tokens in targets]

            meteor = sum([meteor_score([all_targets[i]], preds) for i, preds in enumerate(all_preds) \
                if not len(preds) == 0]) / len(all_targets)
            avg_meteor += meteor

        return avg_meteor / len(list(self.dataloaders[-1]))

    def train(self):
        try:
            with alive_bar(self.trainer_config.num_epochs, 
                           bar='circles', 
                           title="Training:", 
                           title_length=9) as bar:
                for self.epoch in range(1, self.trainer_config.num_epochs):
                    train_loss = self._train_epoch()
                    print(f'epoch {self.epoch} avg_training_loss: {train_loss}')

                    test_loss = self._test_epoch()
                    print(f'{Fore.CYAN}epoch {self.epoch} avg_test_loss:     {test_loss}')

                    if self.early_stopper.early_stop(test_loss):
                        self._save_model()
                        break
                    bar()
        except KeyboardInterrupt:
            print(f'\n{Fore.RED}Training interrupted by user')
                
            self._save_model(interrupted=True, epoch=self.epoch)
            sys.exit(0)
            
        self._save_model()
            
    def _save_model(self, interrupted: bool=False, epoch: int = 0):
        if interrupted:
            filepath = f'./results/{self.run_id}/tbc_checkpoint.pt'
                
            if epoch > 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict()
                    }, filepath)
                print(f'Saved model checkpoint under: {filepath}')
            else:
                print(f'{Fore.RED}Could not save model because it did not train for a minimum of one epoch!')
        else:
            filepath = f'./results/{self.run_id}/final_model.pt'
        
            model_scripted = torch.jit.script(self.model)
            model_scripted.save(filepath)
        
    def _load_model_for_inference(self):
        filepath = f'./results/{self.run_id}/tbc_checkpoint.pt'
        self.model = torch.jit.script(filepath)
        
    def _load_model_for_training(self):
        filepath = f'./results/{self.run_id}/tbc_checkpoint.pt'
        checkpoint = torch.load(filepath)
        return checkpoint
