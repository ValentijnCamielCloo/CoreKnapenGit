#Voor CMD
#manim -pql Manim_Visualizer.py VisualizePointCloud --disable_caching

#Voor Manim Sideview
#, run the add-in, change to new scene, load and wait but it exports automatically mp4 (not prefered)

#manimgl Manim_Visualizer.py VisualizePointCloud
#manimgl Manim_Visualizer.py SquareToCircle
#in Manim compiler 
#checkpoint_paste() #starthiet <Gebruik dit voor tijdelijke scene visualize 

from manim import *
import open3d as o3d
import numpy as np

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(RED, opacity=0.5)
        circle.set_stroke(RED_E, width=1)

        self.add(circle)




