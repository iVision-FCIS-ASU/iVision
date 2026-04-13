import numpy as np
import numpy.typing as npt
from tokenizers import Tokenizer

class BlipTokenizer:
    def __init__(self, tokenizer_path: str, max_length: int = -1):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length

        # BOS: 30522
        # EOS: 102
        # PAD: 0
        self.pad_id = 0
        self.unk_id = 100
        self.cls_id = 101
        self.sep_id = 102
        self.eos_id = 102
        self.bos_id = 30522

    def encode(self, text: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        enc = self.tokenizer.encode(text)

        ids = enc.ids

        # add special tokens
        # ids = [self.cls_id] + ids + [self.sep_id]

        attn = [1] * len(ids)

        # padding
        if self.max_length != -1:
            pad_len = self.max_length - len(ids)
            if pad_len > 0:
                ids += [self.pad_id] * pad_len
                attn += [0] * pad_len
            else:
                ids = ids[:self.max_length]
                attn = attn[:self.max_length]

        return (
            np.array(ids, dtype=np.int64)[None, :],
            np.array(attn, dtype=np.int64)[None, :]
        )