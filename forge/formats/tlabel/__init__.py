"""TLabel format support for Forge.

Provides reading and writing of .tlabel tactile annotation files.
TLabel (https://github.com/liesliy/tlabel) is an open standard for
tactile sensor data with a 14-dimensional semantic schema.
"""

from forge.formats.tlabel.reader import TLabelReader

__all__ = ["TLabelReader"]
