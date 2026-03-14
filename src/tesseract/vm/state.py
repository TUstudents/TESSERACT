from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class VMState:
    registers: Dict[int, Any] = field(default_factory=dict)
    memory: Dict[int, Any] = field(default_factory=dict)
    stack: List[Any] = field(default_factory=list)
    pc: int = 0
    flags: Dict[str, Any] = field(default_factory=dict)
