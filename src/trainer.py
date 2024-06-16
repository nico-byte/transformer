import sys
from typing import List

import torch
import torch.nn as nn
import tokenizers
import matplotlib.pyplot as plt
from evaluate import load as load_metric
from src.seq2seq_transformer import Seq2SeqTransformer
from utils.config import TrainerConfig
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
            param_group["lr"] = self.lr

    def get_lr(self):
        """
        Get the current learning rate.
        """

        return self.optimizer.param_groups[0]["lr"]


class LinearWarmupDecayLRScheduler:
    """
    Implements a learning rate scheduler with linear warmup and decay.
    """

    def __init__(self, optimizer, init_lr, max_lr, n_warmup_steps, total_steps):
        """
        Initialize the scheduler.

        Args:
            optimizer: The optimizer to adjust the learning rate for.
            init_lr (float): The initial learning rate.
            max_lr (float): The maximum learning rate.
            n_warmup_steps (int): The number of warmup steps.
            total_steps (int): The total number of steps.
        """

        self.optimizer = optimizer
        self.n_steps = 0
        self.init_lr = init_lr
        self.max_lr = max_lr
        self.n_warmup_steps = n_warmup_steps
        self.total_steps = total_steps
        self.n_decay_steps = total_steps - n_warmup_steps
        self.warmup_lr_step = (max_lr - init_lr) / n_warmup_steps

    def step(self):
        """
        Update the learning rate for the optimizer.
        """

        self.n_steps += 1

        if self.n_steps < self.n_warmup_steps:
            self.lr = self.init_lr + self.n_steps * self.warmup_lr_step
        else:
            self.lr = (
                self.max_lr
                / self.n_decay_steps
                * (self.n_decay_steps - (self.n_steps - self.n_warmup_steps))
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr

    def get_lr(self):
        """
        Get the current learning rate.
        """

        return self.optimizer.param_groups[0]["lr"]


class EarlyStopper:
    """
    Implements early stopping to prevent overfitting during training.
    """

    def __init__(self, warmup: int = 5, patience: int = 1, min_delta: int = 0):
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
        self.min_validation_loss: float = float("inf")
        self.logger = get_logger("EarlyStopper")

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
            self.logger.info(
                f"{self.counter} epochs without improvement. {self.patience - self.counter} epochs left unless model improves."
            )
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    def __init__(self, device):
        self.logger = get_logger("Trainer")

        self.device = device

        self.train_loss_values = []
        self.test_loss_values = []
        self.learning_rate_values = []
        self.test_loss_steps = []

        self.use_amp = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.dataloaders = {}

    @classmethod
    def new_instance(
        cls,
        model: Seq2SeqTransformer,
        translator,
        train_dataloader,
        test_dataloader,
        val_dataloader,
        tokenizer,
        early_stopper,
        trainer_config,
        device,
        run_id,
    ):
        """
        Creates a new instance of the Trainer class with the provided configuration.

        Args:
            model (Seq2SeqTransformer): The sequence-to-sequence transformer model.
            translator: The translator object used for the model.
            train_dataloader: The training data loader.
            test_dataloader: The test data loader.
            val_dataloader: The validation data loader.
            tokenizer: The tokenizer used for the model.
            early_stopper (EarlyStopper): The early stopping object.
            trainer_config (TrainerConfig): The configuration for the trainer.
            device (torch.device): The device to use for training.
            run_id (str): The ID of the current training run.

        Returns:
            Trainer: A new instance of the Trainer class with the provided configuration.
        """
        trainer = cls(device)

        trainer.model = model.to(device)
        trainer.translator = translator

        trainer.num_epochs = trainer_config.num_epochs

        trainer.dataloaders["train"] = train_dataloader
        trainer.dataloaders["test"] = test_dataloader
        trainer.dataloaders["val"] = val_dataloader

        trainer.current_epoch = 1
        trainer.step_size = int(
            len(list(trainer.dataloaders["train"]))
            / (trainer_config.tgt_batch_size / trainer_config.batch_size)
        )
        trainer.tokenizer = tokenizer
        trainer.early_stopper = early_stopper

        trainer.run_id = run_id

        trainer.criterion = nn.CrossEntropyLoss(ignore_index=3)
        trainer.optim = torch.optim.Adam(
            trainer.model.parameters(),
            lr=trainer_config.learning_rate,
            betas=(0.9, 0.98),
            eps=10e-9,
        )

        init_lr = 2e-6

        if trainer_config.lr_scheduler == "inverse_square_root":
            trainer.scheduler = InverseSquareRootLRScheduler(
                optimizer=trainer.optim,
                init_lr=2e-6,
                max_lr=trainer_config.learning_rate,
                n_warmup_steps=trainer_config.warmup_steps,
            )
        elif trainer_config.lr_scheduler == "linear":
            total_steps = int(
                trainer.num_epochs
                * len(list(trainer.dataloaders["train"]))
                / (trainer_config.tgt_batch_size / trainer_config.batch_size)
            )
            init_lr = 2e-6
            trainer.scheduler = LinearWarmupDecayLRScheduler(
                trainer.optim,
                init_lr=init_lr,
                max_lr=trainer_config.learning_rate,
                n_warmup_steps=trainer_config.warmup_steps,
                total_steps=total_steps,
            )

        trainer.learning_rate_values.append(init_lr)
        trainer.grad_accum = trainer_config.tgt_batch_size > trainer_config.batch_size

        if trainer.grad_accum:
            trainer.accumulation_steps = (
                trainer_config.tgt_batch_size // trainer.dataloaders["train"].batch_size
            )
        else:
            trainer.accumulation_steps = 1

        return trainer

    @classmethod
    def evaluate_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer_path: str,
        val_dataloader: str,
        translator,
        device,
    ):
        """
        Evaluates a trained model checkpoint on the validation dataset.

        Args:
        checkpoint_path (str): The path to the saved model checkpoint.
        tokenizer_path (str): The path to the saved tokenizer.
        val_dataloader (str): The validation data loader.
        translator: The translator object used for the model.
        device (torch.device): The device to use for evaluation.

        Returns:
            Tuple[float, float]: The BLEU and ROUGE scores for the evaluated model.
        """
        trainer = cls(device)

        trainer.model = torch.jit.load(checkpoint_path, map_location=device)
        trainer.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
        trainer.translator = translator

        trainer.dataloaders["val"] = val_dataloader

        bleu, rouge = trainer.evaluate(inference=True)

        return bleu, rouge

    @classmethod
    def continue_training(cls, *args, **kwargs):
        return NotImplementedError

    def _train_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns:
            float: The average training loss for the epoch.
        """

        self.model.train()
        losses = 0
        for batch_idx, (src, tgt) in enumerate(self.dataloaders["train"]):
            tgt = tgt.type(torch.LongTensor)
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = (
                self.translator.create_mask(src, tgt_input)
            )

            with torch.autocast(
                device_type=self.device, dtype=torch.float16, enabled=self.use_amp
            ):
                logits = self.model(
                    src,
                    tgt_input,
                    src_mask,
                    tgt_mask,
                    src_padding_mask,
                    tgt_padding_mask,
                )
                tgt_out = tgt[1:, :]
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                )

            self.scaler.scale(loss).backward()

            if self.grad_accum:
                for param in self.model.parameters():
                    param.grad /= self.accumulation_steps

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optim)
                self.scaler.update()
                self.scheduler.step()
                self.learning_rate_values.append(self.scheduler.get_lr())
                # Reset gradients, for the next accumulated batches
                for param in self.model.parameters():
                    param.grad = None
                self.train_loss_values.append(loss.item())

            losses += loss.item()

        return losses / len(list(self.dataloaders["train"]))

    def _test_epoch(self) -> float:
        """
        Test the model for one epoch.

        Returns:
            float: The average test loss for the epoch.
        """

        self.model.eval()
        losses = 0
        with torch.no_grad():
            for src, tgt in self.dataloaders["test"]:
                tgt = tgt.type(torch.LongTensor)
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:-1, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = (
                    self.translator.create_mask(src, tgt_input)
                )

                with torch.autocast(
                    device_type=self.device, dtype=torch.float16, enabled=self.use_amp
                ):
                    logits = self.model(
                        src,
                        tgt_input,
                        src_mask,
                        tgt_mask,
                        src_padding_mask,
                        tgt_padding_mask,
                    )
                    tgt_out = tgt[1:, :]
                    loss = self.criterion(
                        logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1)
                    )

                losses += loss.item()

        return losses / len(list(self.dataloaders["test"]))

    def evaluate(self, inference: bool = False) -> float:
        """
        Evaluate the model on the validation set and compute the average BLEU and ROUGE scores.

        Args:
            inference (bool, optional): If True, evaluate the model in inference mode without using autocast. Defaults to False.

        Returns:
            Tuple[float, float]: The average BLEU and ROUGE scores on the validation set.
        """
        self.model.eval()
        avg_bleu = 0
        avg_rouge = 0
        bleu = load_metric("bleu")
        rouge = load_metric("rouge")

        with torch.no_grad():
            for batch_idx, (src, tgt) in enumerate(self.dataloaders["val"]):
                self.logger.info(
                    f'Evaluating batch {batch_idx+1}/{len(list(self.dataloaders["val"]))}'
                )
                tgt = tgt.type(torch.LongTensor)
                src = src.to(self.device)
                tgt = tgt.to(self.device)

                tgt_input = tgt[:-1, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = (
                    self.translator.create_mask(src, tgt_input)
                )

                if inference:
                    with torch.no_grad():
                        logits = self.model(
                            src,
                            tgt_input,
                            src_mask,
                            tgt_mask,
                            src_padding_mask,
                            tgt_padding_mask,
                        )
                else:
                    with torch.autocast(
                        device_type=self.device,
                        dtype=torch.float16,
                        enabled=self.use_amp,
                    ):
                        logits = self.model(
                            src,
                            tgt_input,
                            src_mask,
                            tgt_mask,
                            src_padding_mask,
                            tgt_padding_mask,
                        )

                predictions = torch.argmax(logits, dim=-1)
                predictions = predictions.T.cpu().numpy().tolist()
                targets = tgt_input.T.cpu().numpy().tolist()

                all_preds = self.tokenizer.decode_batch(predictions)
                all_targets = self.tokenizer.decode_batch(targets)

                bleu_score = bleu.compute(predictions=all_preds, references=all_targets)
                avg_bleu += bleu_score["bleu"]

                rouge_score = rouge.compute(
                    predictions=all_preds, references=all_targets
                )
                avg_rouge += rouge_score["rougeLsum"]

        avg_bleu /= len(list(self.dataloaders["val"]))
        avg_rouge /= len(list(self.dataloaders["val"]))

        return avg_bleu, avg_rouge

    def train(self):
        """
        Train the model until convergence or early stopping.
        """

        try:
            for epoch in range(self.current_epoch, self.num_epochs + 1):
                start_time = perf_counter()
                train_loss = self._train_epoch()
                self.logger.info(
                    f"epoch {epoch} avg_training_loss: {round(train_loss, 3)} ({round(perf_counter()-start_time, 3)}s)"
                )

                start_time = perf_counter()
                test_loss = self._test_epoch()
                self.logger.info(
                    f"epoch {epoch} avg_test_loss: {round(test_loss, 3)} ({round(perf_counter()-start_time, 3)}s)"
                )

                self.test_loss_values.append(test_loss)
                self.test_loss_steps.append(self.current_epoch * self.step_size)

                self.current_epoch += 1

                self._plot()

                early_stop_true = self.early_stopper.early_stop(epoch, test_loss)
                counter = self.early_stopper.counter

                if counter == 1:
                    self._save_model("best_")

                if early_stop_true:
                    self._save_model("last_")
                    break
        except KeyboardInterrupt:
            self.logger.error("Training interrupted by user")
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

        model_filepath = f"./models/{self.run_id}/{name}checkpoint_scripted.pt"

        model_scripted = torch.jit.script(self.model)
        model_scripted.save(model_filepath)
        self.logger.info(f"Saved model checkpoint to {model_filepath}")

    def _save_model_train(self, name=""):
        """
        Save the model checkpoint for further training.

        Args:
            name: The name of the model checkpoint. Defaults to "".
        """

        model_filepath = f"./models/{self.run_id}/{name}checkpoint.pt"

        torch.save(
            {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                # 'scheduler_state_dict': self.scheduler.state_dict(),
            },
            model_filepath,
        )

        self.logger.info(f"Saved model checkpoint to {model_filepath}")

    def load_model(self):
        """
        Load the model from a checkpoint.
        """

        filepath = f"./models/{self.run_id}/checkpoint.pt"
        self.model = torch.jit.script(filepath)
        self.logger.info(f"Model checkpoint have been loaded from {filepath}")

    def _plot(self):
        """
        Plot the learning rate, training loss, and test loss metrics during training.

        This method creates three separate plots:
        1. Learning Rate: Plots the learning rate values over the training steps, with a vertical line indicating the end of the warmup phase.
        2. Training Loss: Plots the training loss values over the training steps, with a vertical line indicating the end of the warmup phase.
        3. Test Loss: Plots the test loss values over the training epochs, with a vertical line indicating the end of the warmup phase.

        The plots are saved to the `./models/{self.run_id}/metrics/` directory.
        """
        # Plot the learning rate function
        plt.figure(figsize=(8, 6))
        plt.plot(self.learning_rate_values, label="Learning Rate")
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Scheduler")
        plt.grid(True)

        # Add vertical line at the end of warmup phase
        plt.axvline(
            x=self.scheduler.n_warmup_steps,
            color="r",
            linestyle="--",
            label="Warmup End",
        )

        plt.legend()
        plt.savefig(f"./models/{self.run_id}/metrics/learning_rate.png")
        plt.close()  # Clear the current figure

        # Plot the learning rate function
        plt.figure(figsize=(8, 6))
        plt.plot(self.train_loss_values, label="Acutal Train Loss")
        plt.plot(
            self._smooth(scalars=self.train_loss_values, weight=0.9),
            label="Smoothed Train Loss",
        )
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training loss vs. steps")
        plt.grid(True)

        # Add vertical line at the end of warmup phase
        plt.axvline(
            x=self.scheduler.n_warmup_steps,
            color="r",
            linestyle="--",
            label="Warmup End",
        )

        plt.legend()
        plt.savefig(f"./models/{self.run_id}/metrics/train_loss.png")
        plt.close()  # Clear the current figure

        # Plot the learning rate function
        plt.figure(figsize=(8, 6))
        plt.plot(self.test_loss_steps, self.test_loss_values, label="Test Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Test loss vs. steps")
        plt.grid(True)

        # Add vertical line at the end of warmup phase
        plt.axvline(
            x=self.scheduler.n_warmup_steps,
            color="r",
            linestyle="--",
            label="Warmup End",
        )

        plt.legend()
        plt.savefig(f"./models/{self.run_id}/metrics/test_loss.png")
        plt.close()  # Clear the current figure

    @staticmethod
    def _smooth(
        scalars: List[float], weight: float
    ) -> List[float]:  # Weight between 0 and 1
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = (
                last * weight + (1 - weight) * point
            )  # Calculate smoothed value
            smoothed.append(smoothed_val)  # Save it
            last = smoothed_val  # Anchor the last smoothed value

        return smoothed
