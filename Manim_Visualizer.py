#manimgl Manim_Visualizer.py SquareToCircle

from manim import *

class SquareToCircle(Scene):
    def construct(self):
        square = Square()  # Create a square
        circle = Circle()  # Create a circle
        
        # Set attributes for the circle
        circle.set_fill(BLUE, opacity=0.5)
        circle.set_stroke(BLUE_E, width=4)
        
        # Add the square to the scene
        self.play(Create(square))  # Show creation of square
        self.wait(1)                # Wait for 1 second
        self.play(ReplacementTransform(square, circle))  # Transform square into circle
        self.wait(1)                # Wait for 1 second
