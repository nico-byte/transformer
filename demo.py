import gradio as gr
from src.translate import check_device
from utils.demo_model_config import ModelConfig

device = check_device("cpu")

# Initialize model configuration
model_config = ModelConfig(device)

# Set up Gradio theme
theme = gr.themes.Default()

en_examples = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Technology is rapidly changing the way we live and work.",
    "Can you recommend a good restaurant nearby?",
    "Despite the rain, they decided to go for a hike.",
]

de_examples = [
    "Die schnelle braune Katze sprang über den hohen Zaun.",
    "Er spielte den ganzen Tag Videospiele.",
    "Das neue Museum in der Stadt ist einen Besuch wert.",
    "Kannst du mir helfen, dieses Problem zu lösen?",
    "Obwohl sie müde war, arbeitete sie bis spät in die Nacht.",
]


# Build Gradio interface
def t5_model_tab():
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
                    model_id = gr.Textbox(value="t5", visible=False)

                with gr.Row():
                    output = gr.Textbox(label="German Sequence", max_lines=3)

                with gr.Row():
                    trns_btn = gr.Button("Translate")
                    trns_btn.click(
                        fn=model_config.translate,
                        inputs=[seed, model_id],
                        outputs=[output],
                    )
                    gr.ClearButton(components=[seed, output, debug_log])

            with gr.Accordion(label="Examples", open=True):
                gr.Examples(
                    examples=en_examples, inputs=[seed], label="English Sequences"
                )


def custom_model_tab():
    with gr.Tab(label="Custom Model"):
        with gr.Column():
            with gr.Accordion("Debug Log", open=True):
                debug_log = gr.TextArea(label="", lines=7, max_lines=12)

            with gr.Group():
                with gr.Row():
                    model_path_en_de = gr.Textbox(
                        value="./models/en-de-small-v3/en-de-small.pt",
                        max_lines=1,
                        visible=False,
                    )
                    tokenizer_path_en_de = gr.Textbox(
                        value="./models/en-de-small-v3/tokenizer.json",
                        max_lines=1,
                        visible=False,
                    )

                    model_path_de_en = gr.Textbox(
                        value="./models/de-en-small-v2/de-en-small.pt",
                        max_lines=1,
                        visible=False,
                    )
                    tokenizer_path_de_en = gr.Textbox(
                        value="./models/de-en-small-v2/tokenizer.json",
                        max_lines=1,
                        visible=False,
                    )

                    load_custom_en_de_btn = gr.Button("Load custom en-de model")
                    load_custom_en_de_btn.click(
                        fn=model_config.set_custom_model,
                        inputs=[model_path_en_de, tokenizer_path_en_de],
                        outputs=[debug_log],
                    )

                    load_custom_de_en_btn = gr.Button("Load custom de-en model")
                    load_custom_de_en_btn.click(
                        fn=model_config.set_custom_model,
                        inputs=[model_path_de_en, tokenizer_path_de_en],
                        outputs=[debug_log],
                    )

            with gr.Group():
                with gr.Row():
                    seed = gr.Textbox(label="Input Sequence", max_lines=2)
                    model_id = gr.Textbox(value="custom", visible=False)

                with gr.Row():
                    output = gr.Textbox(label="Output Sequence", max_lines=3)

                with gr.Row():
                    trns_btn = gr.Button("Translate")
                    trns_btn.click(
                        fn=model_config.translate,
                        inputs=[seed, model_id],
                        outputs=[output],
                    )
                    gr.ClearButton(components=[seed, output, debug_log])

            with gr.Accordion(label="Examples", open=True):
                gr.Examples(
                    examples=en_examples, inputs=[seed], label="English Sequences"
                )
                gr.Examples(
                    examples=de_examples, inputs=[seed], label="German Sequences"
                )


with gr.Blocks(theme=theme) as demo:
    header = gr.Markdown("# KI in den Life Sciences: Machine Translation Demo")
    line1 = gr.Markdown(
        "by [Nico Fuchs](https://github.com/nico-byte) and [Matthias Laton](https://github.com/20DragonSlayer01)"
    )
    line2 = gr.Markdown("---")
    line3 = gr.Markdown(
        "### This demo uses a T5 model to translate English to German. You can also load your own model and tokenizer."
    )

    t5_model_tab()
    custom_model_tab()

# Launch the Gradio demo
demo.launch()
