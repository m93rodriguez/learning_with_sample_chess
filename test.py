import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
from Piece import Pawn, Rook, Knight, Bishop, Queen, King, Team


def foo(a, b, bar=True, **kwargs):
    if bar:
        return a
    return b


print(foo(b=1, a=2, c=3, d=4))
