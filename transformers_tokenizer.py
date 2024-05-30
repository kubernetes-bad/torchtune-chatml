from typing import List, Optional, Tuple

from torchtune.modules.tokenizers import Tokenizer
from torchtune.data import Message, truncate
from transformers import AutoTokenizer


class TransformersTokenizer(Tokenizer):
    eot_id: Optional[int] = None
    eom_id: Optional[int] = None

    def __init__(self, path: str, eot_token: Optional[str] = None, eom_token: Optional[str] = None, stop_token_ids: Optional[List[str]] = None):
        if stop_token_ids is None:
            stop_token_ids = []

        self.tt_model = AutoTokenizer.from_pretrained(path)
        self.bos_id = self.tt_model.bos_token_id
        self.eos_id = self.tt_model.eos_token_id
        self.pad_id = self.tt_model.pad_token_id

        if eot_token is not None:
            eot_id: int = self.tt_model.encode_tokens_to_ids(eot_token)
            if eot_id is None:
                self.eot_id = eot_id
            else:
                print("TransformersTokenizer: Specified eot_token, but tokenizer returned nothing!"
                      "This is most likely a problem!")

        if eom_token is not None:
            eom_id: int = self.tt_model.encode_tokens_to_ids(eom_token)
            if eom_id is None:
                self.eom_id = eom_id
            else:
                print("TransformersTokenizer: Specified eom_token, but tokenizer returned nothing!"
                      "This is most likely a problem!")

        if stop_token_ids:
            self.stop_tokens = stop_token_ids
        else:
            self.stop_tokens = [self.eos_id]
            if self.eot_id:
                self.stop_tokens.append(self.eot_id)
            if self.eom_id:
                self.stop_tokens.append(self.eom_id)

    def encode(self, text: str, add_bos: bool, add_eos: bool, **kwargs) -> List[int]:
        tokens = self.tt_model.encode(text, **kwargs, add_special_tokens=False)
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tt_model.decode(token_ids, **kwargs)

    def tokenize_messages(self, messages: List[Message], max_seq_len: Optional[int] = None, **kwargs) -> Tuple[List[int], List[bool]]:
        tokens = [self.bos_id]
        # bos and eos are always masked
        mask = [True]
        for message in messages:
            tokenized_message = self._tokenize_message(message)
            tokens = tokens + tokenized_message
            mask = mask + ([message.masked] * len(tokenized_message))
            if max_seq_len and len(tokens) >= max_seq_len:
                break
        if self.eos_id:
            tokens = tokens + [self.eos_id]
        mask = mask + [True]
        if max_seq_len:
            tokens = truncate(tokens, max_seq_len, self.eos_id)
            mask = truncate(mask, max_seq_len, True if self.eos_id else None)
        return tokens, mask

    def _tokenize_message(self, message: Message):
        tokenized_message = self.encode(message.content.strip(), add_bos=False, add_eos=False)
        if message.eot:
            tokenized_message = tokenized_message + [self.eot_id]
        elif self.eom_id:
            tokenized_message = tokenized_message + [self.eom_id]
        return tokenized_message
