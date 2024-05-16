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
                 shared_store):
        self.trainer_config = trainer_config
        self.shared_store = shared_store
        self.model = model
        self.optim = torch.optim.AdamW(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss(ignore_index=shared_store.special_symbols.index('<pad>'))
        STEPSIZE = (len(list(shared_store.dataloaders[0])) /
            (trainer_config.tgt_batch_size / trainer_config.batch_size) * trainer_config.num_epochs) // trainer_config.num_cycles
        
        self.scheduler = CyclicLR(self.optim, base_lr=2e-6, max_lr=trainer_config.learning_rate, mode='triangular2', 
                     step_size_up=STEPSIZE/2, step_size_down=STEPSIZE/2, cycle_momentum=False)
        
        self.early_stopper = early_stopper
        self.run_id = 1234
        
        self.dataloaders = shared_store.dataloaders
        
        self.device = trainer_config.device
        self.grad_accum: bool = trainer_config.tgt_batch_size > trainer_config.batch_size

        if self.grad_accum:
            self.accumulation_steps = trainer_config.tgt_batch_size // self.dataloaders[0].batch_size \
                if trainer_config.tgt_batch_size > self.dataloaders[0].batch_size else 1


    def train_epoch(self) -> float:
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


    def test_epoch(self) -> float:
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
                for epoch in range(self.trainer_config.num_epochs):
                    train_loss = self.train_epoch()
                    print(f'epoch {epoch+1} avg_training_loss: {train_loss}')

                    test_loss = self.test_epoch()
                    print(f'{Fore.CYAN}epoch {epoch+1} avg_test_loss:     {test_loss}')

                    if self.early_stopper.early_stop(test_loss):
                        break
                    bar()
        except KeyboardInterrupt:
            print(f'\n{Fore.RED}Training interrupted by user')
                
            if not os.path.exists(f'./results/{self.run_id}'):
                os.makedirs(f'./results/{self.run_id}')
                
            filepath = f'./results/{self.run_id}/epoch-{epoch}.pt'
                
            if epoch > 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optim.state_dict(),
                    'loss': train_loss,
                    }, filepath)
                print(f'Saved model checkpoint under: {filepath}')
            sys.exit(0)
