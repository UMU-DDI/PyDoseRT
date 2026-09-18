"""Exception types raised by PyDoseRT.

Every error the package raises on its own behalf is one of these, so callers can
catch what PyDoseRT reports without also catching unrelated ``Exception``s, and
so the message can name the offending value and the way out. They all derive from
:class:`PyDoseRTError`.
"""


class PyDoseRTError(Exception):
    """Base class for every error PyDoseRT raises."""


class ShapeError(PyDoseRTError, ValueError):
    """A tensor does not have the shape the pipeline requires.

    Also a ``ValueError``, so existing ``except ValueError`` handlers keep working.
    """


class DeviceDtypeError(PyDoseRTError, ValueError):
    """Tensors disagree on device or dtype, or neither could be determined."""


class EngineStateError(PyDoseRTError, RuntimeError):
    """The engine was asked to do something before it was configured for it."""


class StructureError(PyDoseRTError, KeyError):
    """A requested structure mask is missing or cannot be used."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return self.args[0] if self.args else ""


class GeometryError(PyDoseRTError, ValueError):
    """Beam geometry is inconsistent or unusable."""
