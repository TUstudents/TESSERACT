from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from tesseract.backbone.interface import Backbone, BackboneOutput
from tesseract.compiler.synthetic import (
    RESULT_REGISTER,
    build_arithmetic_prompt,
    build_max_prompt,
    build_sum_to_n_prompt,
)

_INT_PATTERN: Final[str] = r"-?\d+"


@dataclass(frozen=True)
class RuleBasedBackbone(Backbone):
    result_register: int = RESULT_REGISTER

    def encode(
        self,
        prompt: str,
        *,
        repair_hint: str | None = None,
    ) -> BackboneOutput:
        del repair_hint
        normalized = self._normalize(prompt)

        arithmetic_match = self._match_arithmetic(normalized)
        if arithmetic_match is not None:
            operation, lhs, rhs = arithmetic_match
            return BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=build_arithmetic_prompt(operation, lhs, rhs),
                task_type="arithmetic",
                result_register=self.result_register,
                values=(lhs, rhs),
                metadata={"operation": operation},
            )

        max_match = self._match_max(normalized)
        if max_match is not None:
            lhs, rhs = max_match
            return BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=build_max_prompt(lhs, rhs),
                task_type="max",
                result_register=self.result_register,
                values=(lhs, rhs),
            )

        sum_match = self._match_sum_to_n(normalized)
        if sum_match is not None:
            return BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=build_sum_to_n_prompt(sum_match),
                task_type="sum_to_n",
                result_register=self.result_register,
                values=(sum_match,),
            )

        raise ValueError(f"unsupported natural-language prompt {prompt!r}")

    def _normalize(self, prompt: str) -> str:
        lowered = prompt.lower().strip()
        return re.sub(r"\s+", " ", lowered)

    def _match_arithmetic(self, prompt: str) -> tuple[str, int, int] | None:
        patterns: tuple[tuple[str, str], ...] = (
            ("add", rf"(?:what is )?({_INT_PATTERN}) plus ({_INT_PATTERN})\??"),
            ("add", rf"add ({_INT_PATTERN}) and ({_INT_PATTERN})"),
            ("sub", rf"(?:what is )?({_INT_PATTERN}) minus ({_INT_PATTERN})\??"),
            ("sub", rf"subtract ({_INT_PATTERN}) from ({_INT_PATTERN})"),
            ("mul", rf"(?:what is )?({_INT_PATTERN}) times ({_INT_PATTERN})\??"),
            ("mul", rf"multiply ({_INT_PATTERN}) and ({_INT_PATTERN})"),
            ("div", rf"(?:what is )?({_INT_PATTERN}) divided by ({_INT_PATTERN})\??"),
            ("div", rf"divide ({_INT_PATTERN}) by ({_INT_PATTERN})"),
        )
        for operation, pattern in patterns:
            match = re.fullmatch(pattern, prompt)
            if match is None:
                continue
            lhs = int(match.group(1))
            rhs = int(match.group(2))
            if operation == "sub" and prompt.startswith("subtract "):
                return operation, rhs, lhs
            return operation, lhs, rhs
        return None

    def _match_max(self, prompt: str) -> tuple[int, int] | None:
        patterns = (
            rf"max of ({_INT_PATTERN}) and ({_INT_PATTERN})",
            rf"which number is larger: ({_INT_PATTERN}) or ({_INT_PATTERN})\??",
            rf"which is bigger, ({_INT_PATTERN}) or ({_INT_PATTERN})\??",
            rf"compare ({_INT_PATTERN}) and ({_INT_PATTERN}) and return the larger",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, prompt)
            if match is not None:
                return int(match.group(1)), int(match.group(2))
        return None

    def _match_sum_to_n(self, prompt: str) -> int | None:
        patterns = (
            rf"sum integers from 1 to ({_INT_PATTERN})",
            rf"sum numbers from 1 to ({_INT_PATTERN})",
            rf"sum all integers up to ({_INT_PATTERN})",
            rf"compute the triangular number of ({_INT_PATTERN})",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, prompt)
            if match is not None:
                return int(match.group(1))
        return None
