import copy

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.prompts import build_messages

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
MAX_NEW_TOKENS = 128
# Múltiplo do tile de 384 px. O padrão 1536 ampliava as imagens de 900 px, gastando tokens sem ganhar detalhe.
LONGEST_EDGE = 1152


def load_vlm(model_id=MODEL_ID):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    if device == "cuda":
        # GPU sem bf16 (ex.: T4 do Colab) emula e fica lenta; fp16 é nativo nela.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_id)
    processor.image_processor.size = {"longest_edge": LONGEST_EDGE}
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device)
    model.eval()
    return processor, model, device, dtype


def prepare_inputs(processor, device, image_path, prompt_text):
    image = Image.open(image_path).convert("RGB")
    text = processor.apply_chat_template(build_messages(prompt_text), add_generation_prompt=True)
    return processor(text=text, images=[image], return_tensors="pt").to(device)


def generate_answer(processor, model, device, image_path, prompt_text, max_new_tokens=MAX_NEW_TOKENS):
    inputs = prepare_inputs(processor, device, image_path, prompt_text)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    prompt_tokens = inputs["input_ids"].shape[1]
    # generate devolve prompt + resposta.
    new_tokens = output[0][prompt_tokens:]
    return {
        "answer": processor.decode(new_tokens, skip_special_tokens=True).strip(),
        "prompt_tokens": int(prompt_tokens),
        "new_tokens": int(new_tokens.shape[0]),
    }


def common_prefix_length(sequences):
    shortest = min(sequence.shape[0] for sequence in sequences)
    for index in range(shortest):
        if len({int(sequence[index]) for sequence in sequences}) > 1:
            return index
    return shortest


def generate_answers_shared_image(processor, model, device, image_path, prompts, max_new_tokens=MAX_NEW_TOKENS):
    encoded = [prepare_inputs(processor, device, image_path, prompt) for prompt in prompts]
    # A imagem vem antes da pergunta, então a parte cara do prompt é comum e roda uma só vez.
    prefix_length = common_prefix_length([item["input_ids"][0] for item in encoded])
    first = encoded[0]
    with torch.no_grad():
        prefix = model(
            input_ids=first["input_ids"][:, :prefix_length],
            attention_mask=first["attention_mask"][:, :prefix_length],
            pixel_values=first["pixel_values"],
            pixel_attention_mask=first["pixel_attention_mask"],
            use_cache=True,
        )
    results = []
    for item in encoded:
        with torch.no_grad():
            # Recebe o prompt inteiro, mas só processa os tokens depois do cache; por isso dispensa pixel_values.
            output = model.generate(
                input_ids=item["input_ids"],
                attention_mask=item["attention_mask"],
                # generate altera o cache; sem a cópia, uma pergunta contaminaria a seguinte.
                past_key_values=copy.deepcopy(prefix.past_key_values),
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        prompt_tokens = item["input_ids"].shape[1]
        new_tokens = output[0][prompt_tokens:]
        results.append(
            {
                "answer": processor.decode(new_tokens, skip_special_tokens=True).strip(),
                "prompt_tokens": int(prompt_tokens),
                "new_tokens": int(new_tokens.shape[0]),
            }
        )
    return results
