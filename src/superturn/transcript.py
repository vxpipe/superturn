"""Reference packing for the combined model's live partial transcript."""

import unicodedata

TEXT_BYTES = 128


def pack_partial_transcript(partial: str | None) -> list[int]:
    """Return the right-aligned unsigned byte values for one int32 input row."""
    normalized = unicodedata.normalize("NFKC", partial or "").lower()
    text = "".join(char for char in normalized
                   if char != "\0" and not unicodedata.category(char).startswith("P"))
    encoded = text.encode("utf-8")[-TEXT_BYTES:]
    return [0] * (TEXT_BYTES - len(encoded)) + list(encoded)
