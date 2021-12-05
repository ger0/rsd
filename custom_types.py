from enum import Enum
class Colors(Enum):
    BLUE    = 0
    RED     = 1

class Type(Enum):
    parking     = 0
    crosswalk   = 1
    roundabout  = 2
    stop        = 3
