import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import time
from Piece import Match

Game = Match()
fig = Game.initialize()

cid = fig.canvas.mpl_connect('button_press_event', Game.onclick)


plt.show()

