"""The signals the package's models announce their changes with.

Built on psygnal (Qt-free, so the models stay testable without a GUI),
but this is the only module that imports it: everything else gets
`Signal`, `SignalInstance` and the helpers from here, so the choice of
library stays in one place.

Declare a signal on a class -- `changed: ClassVar[Signal] = Signal()` --
and each instance gets its own SignalInstance to connect to and emit.
"""

from __future__ import annotations

from collections.abc import Callable

from psygnal import Signal, SignalInstance

__all__ = ["Signal", "SignalInstance", "disconnect"]


def disconnect(signal: SignalInstance, slot: Callable[..., object]) -> None:
    """signal.disconnect(slot), with the one unavoidable type-check
    suppression in one place: psygnal annotates `slot` as a bare
    `Callable`, which reads as partially unknown."""
    signal.disconnect(slot)  # pyright: ignore[reportUnknownMemberType]
