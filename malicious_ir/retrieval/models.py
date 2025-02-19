import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import numpy as np


class Promptriever:
    def __init__(self, model_name_or_path, max_length=512, torch_dtype=torch.float32, device="cuda"):
        self.model, self.tokenizer = self.get_model(model_name_or_path, max_length=max_length, torch_dtype=torch_dtype)
        self.model.eval().to(device)

    def get_model(self, peft_model_name, max_length=512, torch_dtype=torch.float32):
        # Load the PEFT configuration to get the base model name
        peft_config = PeftConfig.from_pretrained(peft_model_name)
        base_model_name = peft_config.base_model_name_or_path

        # Load the base model and tokenizer
        base_model = AutoModel.from_pretrained(base_model_name, torch_dtype=torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        # Load and merge the PEFT model
        model = PeftModel.from_pretrained(base_model, peft_model_name)
        model = model.merge_and_unload()

        # can be much longer, but for the example 512 is enough
        model.config.max_length = max_length
        tokenizer.model_max_length = max_length

        return model, tokenizer

    def create_batch_dict(self, tokenizer, input_texts):
        max_length = self.model.config.max_length
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def encode(self, sentences, max_length: int = 2048, batch_size: int = 4):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]

            batch_dict = self.create_batch_dict(self.tokenizer, batch_texts)
            batch_dict = {
                key: value.to(self.model.device) for key, value in batch_dict.items()
            }

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    last_hidden_state = outputs.last_hidden_state
                    sequence_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                    batch_size = last_hidden_state.shape[0]
                    reps = last_hidden_state[
                        torch.arange(batch_size, device=last_hidden_state.device),
                        sequence_lengths,
                    ]
                    embeddings = F.normalize(reps, p=2, dim=-1)
                    all_embeddings.append(embeddings)

        return torch.concatenate(all_embeddings, axis=0)
