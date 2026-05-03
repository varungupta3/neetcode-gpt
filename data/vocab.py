from typing import Dict, List, Tuple, Set

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = list(text)
        unique_chars = sorted(set(chars))
        stoi = {}
        itos = {}
        enc_val = 0
        for char in unique_chars:
            stoi[char] = enc_val
            itos[enc_val] = char
            enc_val += 1
        return (stoi, itos)

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        chars = list(text)
        return [stoi[c] for c in chars]

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        return "".join([itos[i] for i in ids])