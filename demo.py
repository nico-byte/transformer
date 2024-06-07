import gradio as gr
import os
import shutil
from src.translate import check_device, translate_sequence_from_checkpoint
from src.t5_inference import get_base_model, t5_inference

device = check_device('cpu')

class ModelConfig:
    def __init__(self, model=None, custom_tokenizer=None, tokenizer=None):
        self.model = model
        self.custom_tokenizer = custom_tokenizer
        self.tokenizer = tokenizer
        
    def set_t5_model(self):
        try:
            tokenizer, model = get_base_model(device)
            self.model, self.tokenizer, self.custom_tokenizer = model, tokenizer, None
            return f"T5 Model loaded: {self.model}, {self.tokenizer}"
        except RuntimeError as e:
            print(e)
            return "Something went wrong!"
        
    def set_custom_model(self, model, tokenizer):
        if not isinstance(model, gr.utils.NamedString) or not isinstance(tokenizer, gr.utils.NamedString):
            return f"Please provide a model and tokenizer, {model}; {tokenizer}"
        model = self.process_file(model)
        tokenizer = self.process_file(tokenizer)
        try:
            self.model, self.custom_tokenizer, self.tokenizer = model, tokenizer, None
            return f"Custom Model loaded: {self.model}, {self.tokenizer}"
        except RuntimeError as e:
            return e
        
    @staticmethod
    def process_file(fileobj):
        if not os.path.exists('./.temps'):
            os.makedirs('./.temps')
        
        path = "./.temps/" + os.path.basename(fileobj)
        shutil.copyfile(fileobj.name, path)
        return path
        
    def _translate_t5(self, sequence):
        try:
            output = t5_inference(self.tokenizer, self.model, sequence, device)
            return output
        except RuntimeError as e:
            return e
        
    def _translate_custom(self, sequence):
        try:
            output = translate_sequence_from_checkpoint(self.model, self.custom_tokenizer, sequence, device)
            return output
        except RuntimeError as e:
            return e
        
    def translate(self, sequence):
        if self.custom_tokenizer is not None:
            return self._translate_custom(sequence)
        elif self.tokenizer is not None:
            return self._translate_t5(sequence)
        else:
            return 'Load the model first!'
        
model_config = ModelConfig()

theme = gr.themes.Default()

with gr.Blocks(theme=theme) as demo:
    header = gr.Markdown("# KI in den Life Sciences: Machine Translation Demo")
    line1 = gr.Markdown("by [Nico Fuchs](https://github.com/nico-byte) and [Matthias Laton](https://github.com/20DragonSlayer01)")
    line2 = gr.Markdown("---")
    line3 = gr.Markdown("### This demo uses a T5 model to translate English to German. You can also load your own model and tokenizer.")
    
    with gr.Tab(label="T5 Model"):
        with gr.Column():
            with gr.Accordion("Debug Log", open=False):
                 debug_log = gr.TextArea(label="Debug Log", lines=7, max_lines=12)

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

    with gr.Tab(label="Custom Model"):
        with gr.Column():
            with gr.Accordion("Debug Log", open=False):
                debug_log = gr.TextArea(label="Debug Log", lines=7, max_lines=12)

            with gr.Group():
                with gr.Row():
                    model = gr.File(label="Model", file_types=['.pt'])
                    tokenizer = gr.File(label="Tokenizer", file_types=['.json'])

                with gr.Row():
                    load_custom_btn = gr.Button("Load custom model")
                    load_custom_btn.click(fn=model_config.set_custom_model, inputs=[model, tokenizer], outputs=[debug_log])

            with gr.Group():
                with gr.Row():
                   seed = gr.Textbox(label="English Sequence", max_lines=2)

                with gr.Row():
                    output = gr.Textbox(label="German Sequence", max_lines=3)

                with gr.Row():
                    trns_btn = gr.Button("Translate")
                    trns_btn.click(fn=model_config.translate, inputs=[seed], outputs=[output])
                    clear_btn = gr.ClearButton(components=[seed, output, debug_log])

demo.launch()