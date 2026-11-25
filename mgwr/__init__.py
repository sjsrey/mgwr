import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import diagnostics as diagnostics
from . import gwr as gwr
from . import kernels as kernels
from . import sel_bw as sel_bw

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("mgwr")
