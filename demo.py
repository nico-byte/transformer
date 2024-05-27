import gradio as gr
import os
import shutil
from translate import check_device, translate_sequence_from_checkpoint
from t5_inference import get_base_model, t5_inference

device = check_device('cpu')

class ModelConfig:
    def __init__(self, model=None, vocab=None, tokenizer=None):
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer
        
    def set_t5_model(self):
        try:
            tokenizer, model = get_base_model(device)
            self.model, self.tokenizer, self.vocab = model, tokenizer, None
            return f"T5 Model loaded: {self.model}, {self.tokenizer}"
        except RuntimeError as e:
            print(e)
            return "Something went wrong!"
        
    def set_custom_model(self, model, vocab):
        model = self.process_file(model)
        vocab = self.process_file(vocab)
        if not model.endswith('.pt') or not vocab.endswith('.pth'):
            return f"Please provide a model and vocab, {model}; {vocab}"
        try:
            self.model, self.vocab, self.tokenizer = model, vocab, None
            return f"Custom Model loaded: {self.model}, {self.vocab}"
        except RuntimeError as e:
            return "Something went wrong!"
        
    @staticmethod
    def process_file(fileobj):
        if not os.path.exists('./.temps'):
            os.makedirs('./.temps')
        
        path = "./.temps/" + os.path.basename(fileobj)
        shutil.copyfile(fileobj.name, path)
        return str(path)
        
    def _translate_t5(self, sequence):
        try:
            output = t5_inference(self.tokenizer, self.model, sequence, device)
            return output
        except RuntimeError as e:
            return e
        
    def _translate_custom(self, sequence):
        try:
            output = translate_sequence_from_checkpoint(self.model, self.vocab, sequence, device)
            return output
        except RuntimeError as e:
            return e
        
    def translate(self, sequence):
        if self.vocab is not None:
            return self._translate_custom(sequence)
        elif self.tokenizer is not None:
            return self._translate_t5(sequence)
        else:
            return 'Load the model first!'
        
model_config = ModelConfig()

with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.amber, secondary_hue=gr.themes.colors.fuchsia, neutral_hue=gr.themes.colors.neutral)) as demo:
    with gr.Column():
        with gr.Accordion("Debug Log"):
            debug_log = gr.TextArea(label="Debug Log", lines=7, max_lines=7)
        with gr.Group():
            with gr.Row():
                model = gr.File(label="Model", file_types=['.pt'])
                vocab = gr.File(label="Vocab", file_types=['.pth'])
        with gr.Row():
            load_custom_btn = gr.Button("Load custom model")
            load_custom_btn.click(fn=model_config.set_custom_model, inputs=[model, vocab], outputs=[debug_log])
            load_t5_btn = gr.Button("Load T5 model")
            load_t5_btn.click(fn=model_config.set_t5_model, outputs=[debug_log])
        with gr.Row():
            seed = gr.Textbox(label="English Sequence", max_lines=2)
        with gr.Row():
            output = gr.Textbox(label="German Sequence", max_lines=3)
        with gr.Row():
            trns_btn = gr.Button("Translate")
            trns_btn.click(fn=model_config.translate, inputs=[seed], outputs=[output])
            cls_btn = gr.ClearButton(components=[seed, output, debug_log])

demo.launch()