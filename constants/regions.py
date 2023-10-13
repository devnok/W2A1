from typing import Final
import numpy as np

# 열대서인도양 region (50E-70E, 10S-0)
REGION_WTIO: Final = np.array([[50, -10], [70, 10]])

# 열대남동인도양 region (90E-110E, 10S-0)
REGION_SETIO: Final = np.array([[90, -10], [110, 0]])

# Nino3.4 region (170W-120W, 5S-5N)
NINO34: Final = np.array([[-170, -5], [-120, 5]])
