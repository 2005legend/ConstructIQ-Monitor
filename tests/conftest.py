import sys
from unittest.mock import MagicMock

# Global mock for pycolmap to allow testing without the binary dependency
sys.modules['pycolmap'] = MagicMock()
