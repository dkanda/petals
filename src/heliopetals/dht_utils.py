import warnings

warnings.warn(
    "heliopetals.dht_utils has been moved to heliopetals.utils.dht. This alias will be removed in Petals 2.2.0+",
    DeprecationWarning,
    stacklevel=2,
)

from heliopetals.utils.dht import *
