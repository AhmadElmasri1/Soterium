import pyrealsense2 as rs
import numpy as np 
import cv2

class CameraHandler:

    
    def __init__(self, name = "Camera 1", type = "D435", width = 1280, height = 720, frameRate = 30):
        self.name = name
        self.type = type
        self.width = width
        self.height = height
        self.frameRate = frameRate
        self.pipeline = None

    def start(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.frameRate)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.frameRate)

        # Start streaming
        self.pipeline.start(config)

    def getColorFrame(self):
        frames = self.pipeline.wait_for_frames()
        colorFrame = frames.get_color_frame()
        colorImage = np.asanyarray(colorFrame.get_data())
        return colorImage

    def getDepthFrame(self):
        frames = self.pipeline.wait_for_frames()
        depthFrame = frames.get_depth_frame()
        depthImage = np.asanyarray(depthFrame.get_data())
        depthColorMap = cv2.applyColorMap(cv2.convertScaleAbs(depthImage, alpha=0.03), cv2.COLORMAP_JET)
        return depthColorMap
