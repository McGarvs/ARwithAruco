import pyglet
import os
FILE_PATH = os.path.join(os.getcwd(), "RR.mp4")
player = pyglet.media.Player()
MediaLoad = pyglet.media.load(FILE_PATH)
window = pyglet.window.Window()
player.queue(MediaLoad)
player.play()
player.volume = 0.005
pyglet.app.run()
