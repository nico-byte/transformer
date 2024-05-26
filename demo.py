import gradio as gr
from translate import check_device
from t5_inference import get_base_model, t5_inference

device = check_device('cpu')

tokenizer, model = get_base_model(device)

def translate(sequence):
    output = t5_inference(tokenizer, model, sequence, device)
    return output

demo = gr.Interface(
    fn=translate,
    inputs=[gr.Textbox(label="Input", lines=1, info="Enter an english sentence.")],
    outputs=[gr.Textbox(label="Output", lines=1, info="The translated sentence in german.")],
)

demo.launch()