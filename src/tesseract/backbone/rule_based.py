from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from tesseract.backbone.interface import Backbone, BackboneOutput
from tesseract.compiler.synthetic import (
    RESULT_REGISTER,
    build_abs_prompt,
    build_arithmetic_prompt,
    build_factorial_prompt,
    build_fibonacci_prompt,
    build_max_prompt,
    build_memory_sum_prompt,
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
        normalized = self._normalize(prompt)
        metadata = self._base_metadata(repair_hint)

        arithmetic_match = self._match_arithmetic(normalized)
        if arithmetic_match is not None:
            operation, lhs, rhs = arithmetic_match
            return BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=build_arithmetic_prompt(operation, lhs, rhs),
                task_type="arithmetic",
                result_register=self.result_register,
                values=(lhs, rhs),
                metadata={**metadata, "operation": operation},
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
                metadata=metadata,
            )

        sum_match = self._match_sum_to_n(normalized)
        if sum_match is not None:
            return BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=build_sum_to_n_prompt(sum_match),
                task_type="sum_to_n",
                result_register=self.result_register,
                values=(sum_match,),
                metadata=metadata,
            )

        factorial_match = self._match_factorial(normalized)
        if factorial_match is not None:
            return BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=build_factorial_prompt(factorial_match),
                task_type="factorial",
                result_register=self.result_register,
                values=(factorial_match,),
                metadata=metadata,
            )

        fibonacci_match = self._match_fibonacci(normalized)
        if fibonacci_match is not None:
            return BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=build_fibonacci_prompt(fibonacci_match),
                task_type="fibonacci",
                result_register=self.result_register,
                values=(fibonacci_match,),
                metadata=metadata,
            )

        abs_match = self._match_abs(normalized)
        if abs_match is not None:
            return BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=build_abs_prompt(abs_match),
                task_type="abs",
                result_register=self.result_register,
                values=(abs_match,),
                metadata=metadata,
            )

        memory_sum_match = self._match_memory_sum(normalized)
        if memory_sum_match is not None:
            return BackboneOutput(
                original_prompt=prompt,
                canonical_prompt=build_memory_sum_prompt(memory_sum_match),
                task_type="memory_sum",
                result_register=self.result_register,
                values=memory_sum_match,
                metadata=metadata,
            )

        raise ValueError(f"unsupported natural-language prompt {prompt!r}")

    def _base_metadata(self, repair_hint: str | None) -> dict[str, int | float | str | bool]:
        if repair_hint is None:
            return {}
        return {"repair_hint": repair_hint}

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

    def _match_factorial(self, prompt: str) -> int | None:
        patterns = (
            rf"factorial of ({_INT_PATTERN})",
            rf"compute factorial of ({_INT_PATTERN})",
            rf"what is the factorial of ({_INT_PATTERN})\??",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, prompt)
            if match is not None:
                return int(match.group(1))
        return None

    def _match_fibonacci(self, prompt: str) -> int | None:
        patterns = (
            rf"fibonacci of ({_INT_PATTERN})",
            rf"compute fibonacci of ({_INT_PATTERN})",
            rf"what is fibonacci of ({_INT_PATTERN})\??",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, prompt)
            if match is not None:
                return int(match.group(1))
        return None

    def _match_abs(self, prompt: str) -> int | None:
        patterns = (
            rf"absolute value of ({_INT_PATTERN})",
            rf"compute absolute value of ({_INT_PATTERN})",
            rf"return the absolute value of ({_INT_PATTERN})",
        )
        for pattern in patterns:
            match = re.fullmatch(pattern, prompt)
            if match is not None:
                return int(match.group(1))
        return None

    def _match_memory_sum(self, prompt: str) -> tuple[int, ...] | None:
        prefixes = (
            "sum memory values ",
            "compute memory sum ",
            "add the memory cells ",
        )
        for prefix in prefixes:
            if prompt == prefix.strip():
                return ()
            if prompt.startswith(prefix):
                suffix = prompt[len(prefix) :]
                tokens = suffix.replace(",", " ").split()
                if tokens and all(re.fullmatch(_INT_PATTERN, token) for token in tokens):
                    values = tuple(int(token) for token in tokens)
                    return values
        return None
