import gradio as gr
import os
import shutil
from src.processor import Processor
from src.translate import check_device, translate_sequence_from_checkpoint
from src.pretrained_inference import get_base_model, t5_inference

device = check_device('cpu')

class ModelConfig:
    """
    A class to manage and configure different translation models.

    Attributes:
        model (str): The model to be used for translation.
        custom_tokenizer (str): The custom tokenizer to be used for translation.
        tokenizer (str): The tokenizer to be used for T5 model translation.
    """

    def __init__(self):
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
        
    def set_t5_model(self):
        """
        Load and set the T5 model and tokenizer.

        Returns:
            str: A message indicating whether the T5 model was successfully loaded.
        """
        try:
            tokenizer, model = get_base_model(device)
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
        if not isinstance(model, gr.utils.NamedString) or not isinstance(tokenizer, gr.utils.NamedString):
            return f"Please provide a model and tokenizer, {model}; {tokenizer}"
        try:
            self.custom_translator = Processor.from_checkpoint(model, tokenizer, device)
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
            output = t5_inference(self.t5_tokenizer, self.t5_model, sequence, device)
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
        
    def translate(self, sequence):
        """
        Translate a sequence using the loaded model.

        Args:
            sequence (str): The input sequence to be translated.

        Returns:
            str: The translated sequence.
        """

        if self.t5_tokenizer and self.t5_model is not None:
            return self._translate_t5(sequence)
        elif self.custom_translator is not None:
            return self._translate_custom(sequence)
        else:
            return 'Load the model first!'

# Initialize model configuration
model_config = ModelConfig()

# Set up Gradio theme
theme = gr.themes.Default()

# Build Gradio interface
with gr.Blocks(theme=theme) as demo:
    header = gr.Markdown("# KI in den Life Sciences: Machine Translation Demo")
    line1 = gr.Markdown("by [Nico Fuchs](https://github.com/nico-byte) and [Matthias Laton](https://github.com/20DragonSlayer01)")
    line2 = gr.Markdown("---")
    line3 = gr.Markdown("### This demo uses a T5 model to translate English to German. You can also load your own model and tokenizer.")
    
    with gr.Tab(label="T5 Model"):
        with gr.Column():
            with gr.Accordion("Debug Log", open=True):
                 debug_log = gr.TextArea(label="", lines=7, max_lines=12)

            with gr.Group():
                load_t5_btn = gr.Button("Load T5 model")
                load_t5_btn.click(fn=model_config.set_t5_model, outputs=[debug_log])

                with gr.Group():
                    with gr.Row():
                       seed = gr.Textbox(label="English Sequence", max_lines=2)

                with gr.Row():
                    output = gr.Textbox(label="German Sequence", max_lines=3)

                with gr.Row():
                    trns_btn = gr.Button("Translate")
                    trns_btn.click(fn=model_config.translate, inputs=[seed], outputs=[output])
                    clear_btn = gr.ClearButton(components=[seed, output, debug_log])
                    
        with gr.Accordion(label="Examples", open=True):
            gr.Examples(examples=[
                "The quick brown fox jumps over the lazy dog.", 
                "She sells seashells by the seashore.", 
                "Technology is rapidly changing the way we live and work.", 
                "Can you recommend a good restaurant nearby?", 
                "Despite the rain, they decided to go for a hike."], 
                        inputs=[seed], label="English Sequences")

    with gr.Tab(label="Custom Model"):
        with gr.Column():
            with gr.Accordion("Debug Log", open=True):
                debug_log = gr.TextArea(label="", lines=7, max_lines=12)

            with gr.Group():
                with gr.Row():
                    model = gr.File(label="Model", file_types=['.pt'], min_width=200)
                    tokenizer = gr.File(label="Tokenizer", file_types=['.json'], min_width=200)

                with gr.Row():
                    load_custom_btn = gr.Button("Load custom model")
                    load_custom_btn.click(fn=model_config.set_custom_model, inputs=[model, tokenizer], outputs=[debug_log])

            with gr.Group():
                with gr.Row():
                   seed = gr.Textbox(label="Input Sequence", max_lines=2)

                with gr.Row():
                    output = gr.Textbox(label="Output Sequence", max_lines=3)

                with gr.Row():
                    trns_btn = gr.Button("Translate")
                    trns_btn.click(fn=model_config.translate, inputs=[seed], outputs=[output])
                    clear_btn = gr.ClearButton(components=[seed, output, debug_log])
        
        with gr.Accordion(label="Examples", open=True):
            gr.Examples(examples=[
                "The quick brown fox jumps over the lazy dog.", 
                "She sells seashells by the seashore.", 
                "Technology is rapidly changing the way we live and work.", 
                "Can you recommend a good restaurant nearby?", 
                "Despite the rain, they decided to go for a hike."], 
                        inputs=[seed], label="English Sequences")
            
            gr.Examples(examples=[
                "Die schnelle braune Katze sprang über den hohen Zaun.", 
                "Er spielte den ganzen Tag Videospiele.", 
                "Das neue Museum in der Stadt ist einen Besuch wert.", 
                "Kannst du mir helfen, dieses Problem zu lösen?", 
                "Obwohl sie müde war, arbeitete sie bis spät in die Nacht."], 
                        inputs=[seed], label="German Sequences")

# Launch the Gradio demo
demo.launch()