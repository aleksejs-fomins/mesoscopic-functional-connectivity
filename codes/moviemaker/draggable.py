import numpy as np
import matplotlib.pyplot as plt

class DraggableCircle:
    def __init__(self, circle, triggerOnRelease=None, triggerOnRightClick=None):
        self.circle = circle
        self.press_shift = None
        self.triggerOnRelease = triggerOnRelease
        self.triggerOnRightClick = triggerOnRightClick

        #connect to all the events we need
        self.cidpress   = self.circle.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.circle.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion  = self.circle.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    # on button press we will see if the mouse is over us and store some data
    def on_press(self, event):
        if event.inaxes != self.circle.axes:
            return

        contains, attrd = self.circle.contains(event)
        if not contains:
            return
        
        # print('event contains', self.circle.center)
        self.press_shift = np.array(self.circle.center) - np.array((event.xdata, event.ydata))

    # on motion we will move the circle if the mouse is over us
    def on_motion(self, event):
        if (self.press_shift is None) or (event.inaxes != self.circle.axes):
            return
        
        self.circle.center = tuple(np.array((event.xdata, event.ydata)) + self.press_shift)
        #self.circle.figure.canvas.draw()

    # on release we reset the press data
    def on_release(self, event):  
        if self.press_shift is not None:
            self.press_shift = None
            if event.button==1:
                print("New coordinates are", self.circle.center)
                if self.triggerOnRelease is not None:
                    self.triggerOnRelease(self.circle.center)
                self.circle.figure.canvas.draw()
            elif event.button==3 and self.triggerOnRightClick is not None:
                print("Setting point as bad")
                # Double-click event - special reaction
                self.triggerOnRightClick(self.circle.center)
                self.circle.figure.canvas.draw()

    # disconnect all the stored connection ids
    def disconnect(self):
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)
