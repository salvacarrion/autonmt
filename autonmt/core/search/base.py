from abc import ABC, abstractmethod


class BaseSearch(ABC):
    """Contract for decoding strategies (greedy, beam, top-k, nucleus, ...).

    Implementations should return ``(token_id_lists, optional_scores)``:
    a list of decoded sequences (one per input) and an optional tensor / list
    of associated scores (or ``None`` if the strategy does not produce them).
    """

    @abstractmethod
    def decode(self, model, dataset, sos_id, eos_id, pad_id, batch_size,
               max_tokens, max_len_a, max_len_b, num_workers, **kwargs):
        ...
