import torch
import transformers
from tqdm import tqdm


@torch.no_grad()
def generate_responses(
    prompts: list[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    max_new_tokens: int = 32,
    num_return_sequences: int = 1,
    batch_size: int = 1,
    show_progress_bar: bool = False,
) -> list[str]:
    data_loader = torch.utils.data.DataLoader(
        prompts, batch_size=batch_size, shuffle=False
    )

    responses = []
    for batch in tqdm(
        data_loader,
        disable=not show_progress_bar,
        leave=False,
        desc="Response generation",
    ):
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        # Assumes the model's generation configuration has already been updated with
        # the desired parameters.
        generated_token_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
        )
        # Only decode the new tokens.
        generated_token_ids = generated_token_ids[:, inputs["input_ids"].shape[1] :]
        responses.extend(
            tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        )

    return responses


if __name__ == "__main__":
    model_name_or_path = "HuggingFaceH4/tiny-random-LlamaForCausalLM"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = [
        "The capital of Canada is",
        "The capital of France is",
        "The capital of Poland is",
        "The capital of Mexico is",
    ]

    responses = generate_responses(prompts, model, tokenizer)
    print(responses)
