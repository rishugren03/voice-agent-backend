"""Tracks token and audio usage per session for the cost breakdown."""
from dataclasses import dataclass, field, asdict


# Pricing as of 2025 — update if rates change
_GPT4O_INPUT_PER_TOKEN = 2.50 / 1_000_000
_GPT4O_OUTPUT_PER_TOKEN = 10.00 / 1_000_000
_DEEPGRAM_PER_MINUTE = 0.0043
_CARTESIA_PER_CHAR = 0.000015


@dataclass
class CostTracker:
    input_tokens: int = 0
    output_tokens: int = 0
    stt_seconds: float = 0.0
    tts_chars: int = 0

    def add_llm_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def add_stt_seconds(self, seconds: float) -> None:
        self.stt_seconds += seconds

    def add_tts_chars(self, chars: int) -> None:
        self.tts_chars += chars

    def to_dict(self) -> dict:
        llm_usd = (
            self.input_tokens * _GPT4O_INPUT_PER_TOKEN
            + self.output_tokens * _GPT4O_OUTPUT_PER_TOKEN
        )
        stt_usd = (self.stt_seconds / 60) * _DEEPGRAM_PER_MINUTE
        tts_usd = self.tts_chars * _CARTESIA_PER_CHAR

        return {
            "llm_usd": round(llm_usd, 5),
            "stt_usd": round(stt_usd, 5),
            "tts_usd": round(tts_usd, 5),
            "total_usd": round(llm_usd + stt_usd + tts_usd, 5),
            "detail": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "stt_seconds": round(self.stt_seconds, 1),
                "tts_chars": self.tts_chars,
            },
        }
