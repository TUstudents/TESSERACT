from __future__ import annotations

import pytest

from tesseract.vm.machine import VM
from tesseract.vm.state import VMState


@pytest.fixture
def vm() -> VM:
    return VM()


@pytest.fixture
def empty_state() -> VMState:
    return VMState()
