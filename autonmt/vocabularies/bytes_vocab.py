from autonmt.vocabularies.base_vocab import BaseVocabulary


class BytesVocabulary(BaseVocabulary):
    def __init__(self, sos_id=256, eos_id=257, pad_id=258,
                 sos_piece="<s>", eos_piece="</s>", pad_piece="<pad>", hex_input=False, lang=None, max_tokens=None):
        super().__init__(sos_id=sos_id, eos_id=eos_id, pad_id=pad_id,
                         sos_piece=sos_piece, eos_piece=eos_piece, pad_piece=pad_piece,
                         lang=lang, max_tokens=max_tokens)
        # Set special tokens
        self._offset = len(self.special_tokens)

        # Other
        self.hex_input = hex_input

    def __len__(self):
        return 256 + len(self.special_tokens)

    def encode(self, text, add_special_tokens=True):
        if self.hex_input:
            b_list = [int(x, base=16) for x in text.split(' ')]
        else:
            s_bytes = text.encode()  # b'Hello world! \xf0\x9f\x8c\xb1'
            b_list = [b for b in s_bytes]  # [72, 101, 108,...]
        idxs = b_list[:self.max_tokens - 2 * int(add_special_tokens)] if self.max_tokens else b_list  # count <sos> and <eos>
        idxs = [self.sos_id] + idxs + [self.eos_id] if add_special_tokens else b_list
        return idxs

    def decode(self, idxs, remove_special_tokens=True, encoding="utf-8"):
        # Remove special tokens
        if remove_special_tokens:
            try:
                # Remove <sos>
                sos_pos = idxs.index(self.sos_id)
                idxs = idxs[sos_pos+1:]
            except ValueError:
                pass
            try:
                # Remove <eos>
                eos_pos = idxs.index(self.eos_id)
                idxs = idxs[:eos_pos]
            except ValueError:
                pass

        # Decode idxs
        if self.hex_input:
            text = " ".join([hex(x) for x in idxs])
        else:
            b_enc = bytes(idxs)
            text = b_enc.decode(encoding, errors="replace")
        return text
