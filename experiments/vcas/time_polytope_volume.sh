#!/bin/bash
python3 -m timeit \
  -s "import numpy as np; from scipy.spatial import ConvexHull" \
  "X=np.random.rand(1000, $n); v=ConvexHull(X).volume; assert v >= 0"
