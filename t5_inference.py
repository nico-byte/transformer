from torchtext.models import T5_BASE_GENERATION
from torchtext.prototype.generate import GenerationUtils


def get_base_model():
    t5_base = T5_BASE_GENERATION
    transform = t5_base.transform()
    model = t5_base.get_model()
    
    sequence_generator = GenerationUtils(model)
    
    return model, transform, sequence_generator


def t5_inference(model, transform, sequence_generator, sequence, device):
    model.to(device)
    model.eval()

    sequence = ["translate English to German: " + sequence]
    model_input = transform(sequence)
    model_output = sequence_generator.generate(model_input, eos_idx=1, num_beams=1)
    output_sequence = transform.decode(model_output.tolist())

    return output_sequence[0]