import torch
from torch import Tensor


class Tensorizer:
    def __init__(self, tokenizer, max_length: int = 256, pad_to_max: bool = True) -> None:
        self.max_length = max_length
        self.pad_to_max = pad_to_max
        self.tokenizer = tokenizer

    def text_to_tensor(self, text, add_special_tokens: bool = True, apply_max_len: bool = True):
        token_ids = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=True,
                truncation=True,
                return_tensors="pt"
            )
        return dict(token_ids)
    
    def get_pair_separator_ids(self) -> Tensor:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: Tensor) -> Tensor:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]