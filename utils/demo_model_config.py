import gradio as gr
from src.processor import Processor
from src.pretrained_inference import get_base_model, t5_inference


class ModelConfig:
    """
    A class to manage and configure different translation models.

    Attributes:
        model (str): The model to be used for translation.
        custom_tokenizer (str): The custom tokenizer to be used for translation.
        tokenizer (str): The tokenizer to be used for T5 model translation.
    """

    def __init__(self, device):
        """
        Initialize the ModelConfig class.

        Args:
            model (str): The model to be used for translation.
            custom_tokenizer (str): The custom tokenizer to be used for translation.
            tokenizer (str): The tokenizer to be used for T5 model translation.
        """

        self.t5_model = None
        self.t5_tokenizer = None
        self.custom_translator = None
        self.device = device

    def set_t5_model(self):
        """
        Load and set the T5 model and tokenizer.

        Returns:
            str: A message indicating whether the T5 model was successfully loaded.
        """
        try:
            tokenizer, model = get_base_model(self.device)
            self.t5_model, self.t5_tokenizer = model, tokenizer
            self.custom_translator = None
            return f"T5 Model loaded: {self.t5_model}, {self.t5_tokenizer}"
        except RuntimeError as e:
            print(e)
            return "Something went wrong!"

    def set_custom_model(self, model, tokenizer):
        """
        Load and set a custom model and tokenizer.

        Args:
            model: The file path to the custom model.
            tokenizer: The file path to the custom tokenizer.

        Returns:
            str: A message indicating whether the custom model was successfully loaded.
        """
        if not isinstance(model, gr.utils.NamedString) or not isinstance(
            tokenizer, gr.utils.NamedString
        ):
            return f"Please provide a model and tokenizer, {model}; {tokenizer}"
        try:
            model, tokenizer = self.process_file(model), self.process_file(tokenizer)
            self.custom_translator = Processor.from_checkpoint(
                model, tokenizer, self.device
            )
            self.t5_model, self.t5_tokenizer = None, None
            return f"Custom Model loaded: {self.custom_translator}"
        except RuntimeError as e:
            print(e)
            return "Something went wrong!"

    @staticmethod
    def process_file(fileobj):
        """
        Process the uploaded file.

        Args:
            fileobj: The file object to be processed.

        Returns:
            str: The file name.
        """

        return fileobj.name

    def _translate_t5(self, sequence):
        """
        Translate a sequence using the T5 model.

        Args:
            sequence (str): The input sequence to be translated.

        Returns:
            str: The translated sequence.
        """

        try:
            output = t5_inference(
                self.t5_tokenizer, self.t5_model, sequence, self.device
            )
            return output
        except RuntimeError as e:
            return e

    def _translate_custom(self, sequence):
        """
        Translate a sequence using the custom model.

        Args:
            sequence (str): The input sequence to be translated.

        Returns:
            str: The translated sequence.
        """

        try:
            output = self.custom_translator.translate(sequence)
            return output
        except RuntimeError as e:
            return e

    def translate(self, sequence, model_id: str):
        """
        Translate a sequence using the loaded model.

        Args:
            sequence (str): The input sequence to be translated.

        Returns:
            str: The translated sequence.
        """

        if self.t5_tokenizer and self.t5_model is not None and model_id == "t5":
            return self._translate_t5(sequence)
        elif self.custom_translator is not None and model_id == "custom":
            return self._translate_custom(sequence)
        else:
            return "Load the model first!"
