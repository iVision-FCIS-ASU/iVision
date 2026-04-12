import numpy as np
from tokenizers import Tokenizer

class ClipTokenizer:
    def __init__(self, tokenizer_path: str, max_length: int = 77):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length

        # CLIP special tokens (important)
        self.bos_token_id = 49406
        self.eos_token_id = 49407
        # self.pad_token_id = 0
        self.pad_token_id = self.eos_token_id

    def encode(self, texts: list[str]):
        input_ids = []
        attention_masks = []

        for text in texts:
            encoding = self.tokenizer.encode(text, add_special_tokens=False)

            ids = encoding.ids

            # Add BOS/EOS manually (CLIP requires this)
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

            # Truncate
            ids = ids[:self.max_length]

            # Attention mask
            attn_mask = [1] * len(ids)

            # Pad
            pad_len = self.max_length - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                attn_mask = attn_mask + [0] * pad_len

            input_ids.append(ids)
            attention_masks.append(attn_mask)

        return (
            np.array(input_ids, dtype=np.int64),
            np.array(attention_masks, dtype=np.int64)
        )