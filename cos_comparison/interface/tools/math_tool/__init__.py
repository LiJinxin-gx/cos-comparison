"""
Provides some advanced math tools.
"""

try:
    from . import _fourier as fourier
except:
    from . import fourier

try:
    from . import _topology as topology
except:
    from . import topology

try:
    from . import _linear_algebra as linear_algebra
except:
    from . import linear_algebra

try:
    from . import _unit_map as unit_map
except:
    from . import unit_map
