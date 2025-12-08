# Copyright (c) 2025 Yihang Sun. Created for the Efficient LVSM project (ICLR 2026).
#
# REPA (REPresentation Alignment) Configuration
# 
# This file defines layer-feature mapping configurations for representation alignment.
# Each configuration specifies which intermediate features from which layers should be
# used for alignment between input and target representations.
#
# Format: 'layer-feature' -> {input: {layer: [features]}, target: {layer: [features]}}
# Example: '12-3' means using feature 3 from layer 12 for both input and target

repa_map = dict()

# Basic configurations - single layer, single feature
repa_map['12-3'] = {
    'input': {
        12: [3]
    },
    'target': {
        12: [3]
    }
}

repa_map['12-4'] = {
    'input': {
        12: [4]
    },
    'target': {
        12: [4]
    }
}

repa_map['8-3'] = {
    'input': {
        8: [3]
    },
    'target': {
        8: [3]
    }
}

# Multiple features from same layer
repa_map['12-34'] = {
    'input': {
        12: [3, 4]
    },
    'target': {
        12: [3, 4]
    }
}

# Target-only configurations
repa_map['12-3t'] = {
    'input': {},
    'target': {
        12: [3]
    }
}

# Input-only configurations
repa_map['12-3i'] = {
    'input': {
        12: [3]
    },
    'target': {}
}

repa_map['10-3'] = {
    'input': {
        10: [3]
    },
    'target': {
        10: [3]
    }
}

repa_map['6-3'] = {
    'input': {
        6: [3]
    },
    'target': {
        6: [3]
    }
}

repa_map['6-4'] = {
    'input': {
        6: [4]
    },
    'target': {
        6: [4]
    }
}

repa_map['8-4'] = {
    'input': {
        8: [4]
    },
    'target': {
        8: [4]
    }
}

repa_map['8-34'] = {
    'input': {
        8: [3,4]
    },
    'target': {
        8: [3,4]
    }
}

repa_map['8-2'] = {
    'input': {
        8: [2]
    },
    'target': {
        8: [2]
    }
}

repa_map['8-2t'] = {
    'input': {},
    'target': {
        8: [2]
    }
}
repa_map['8-2i'] = {
    'input': {
        8: [2]
    },
    'target': {}
}

repa_map['6-2t'] = {
    'input': {},
    'target': {
        6: [2]
    }
}

repa_map['8-1t'] = {
    'input': {},
    'target': {
        8: [1]
    }
}

repa_map['8-2'] = {
    'input': {
        8: [2],
    },
    'target': {
        8: [2]
    }
}

repa_map['10-2'] = {
    'input': {
        10: [2],
    },
    'target': {
        10: [2]
    }
}

repa_map['8-3t-8-2i'] = {
    'input': {
        8: [2],
    },
    'target': {
        8: [3]
    }
}

repa_map['8-e2'] = {
    'input': {
        8: [102],
    },
    'target': {}
}

repa_map['8-3t-8-e4'] = {
    'input': {
        8: [104],
    },
    'target': {
        8: [3]
    }
}
repa_map['8-3t-8-1i'] = {
    'input': {
        8: [1],
    },
    'target': {
        8: [3]
    }
}

# Multi-layer configurations
repa_map['48-23'] = {
    'input': {
        4: [2],
        8: [3]
    },
    'target': {
        4: [2],
        8: [3]
    }
}

repa_map['48-24'] = {
    'input': {
        4: [2],
        8: [4]
    },
    'target': {
        4: [2],
        8: [4]
    }
}

repa_map['48-13'] = {
    'input': {
        4: [1],
        8: [3]
    },
    'target': {
        4: [1],
        8: [3]
    }
}

# Three-layer configurations
repa_map['4812-123'] = {
    'input': {
        4: [2],
        8: [4],
        12: [3]
    },
    'target': {
        4: [2],
        8: [4],
        12: [3]
    }
}

repa_map['369-123'] = {
    'input': {
        3: [1],
        6: [2],
        9: [3]
    },
    'target': {
        3: [1],
        6: [2],
        9: [3]
    }
}