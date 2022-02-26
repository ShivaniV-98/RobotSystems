import sys
import cv2
import math
import rospy
import numpy as np
import threading
import Camera
from LABConfig import *
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Board as Board
from CameraCalibration.CalibrationConfig import *
AK = ArmIK()

range_rgb = {
    'red': (0, 0, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}
def setBuzzer(timer):
    Board.setBuzzer(0)
    Board.setBuzzer(1)
    time.sleep(timer)
    Board.setBuzzer(0)

def set_rgb(color):
    if color == "red":
        Board.RGB.setPixelColor(0, Board.PixelColor(255, 0, 0))
        Board.RGB.setPixelColor(1, Board.PixelColor(255, 0, 0))
        Board.RGB.show()
    elif color == "green":
        Board.RGB.setPixelColor(0, Board.PixelColor(0, 255, 0))
        Board.RGB.setPixelColor(1, Board.PixelColor(0, 255, 0))
        Board.RGB.show()
    elif color == "blue":
        Board.RGB.setPixelColor(0, Board.PixelColor(0, 0, 255))
        Board.RGB.setPixelColor(1, Board.PixelColor(0, 0, 255))
        Board.RGB.show()
    else:
        Board.RGB.setPixelColor(0, Board.PixelColor(0, 0, 0))
        Board.RGB.setPixelColor(1, Board.PixelColor(0, 0, 0))
        Board.RGB.show()
def getAreaMaxContour(contours):
    contour_area_temp = 0
    contour_area_max = 0
    area_max_contour = None

    for c in contours:  # traversal all the contours
        contour_area_temp = math.fabs(cv2.contourArea(c))  # calculate the countour area
        if contour_area_temp > contour_area_max:
            contour_area_max = contour_area_temp
            if contour_area_temp > 300:  # only when the area is greater than 300, the contour of the maximum area is effective to filter interference
                area_max_contour = c

    return area_max_contour, contour_area_max  # return the maximum area countour

def initMove():
    Board.setBusServoPulse(1, servo1 - 50, 300)
    Board.setBusServoPulse(2, 500, 500)
    AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)


class Perception:
    def __init__(self, img):
        self._img = img
        self._imgCopy = img.copy()
        self._imgH, self._imgW = img.shape[:2]

        self._maxArea = 0
        self._maxAreaContour = 0
        self.roi = ()
        self.rect = None
        self.count = 0
        self.get_roi = False
        self.center_list = []
        self.unreachable = False
        self.isRunning = True
        self.start_pick_up = False
        self.rotation_angle = 0
        self.last_x = 0
        self.last_y = 0
        self.world_X = 0
        self.world_Y = 0
        self.start_count_t1 = True
        self.t1 = 0
        self.detect_color = 'None'
        self.draw_color = range_rgb["black"]
        self.color_list = []
        self.size = (640, 480)
        self.target_color = ('red', 'green', 'blue')

    def preprocess(self, img):
        img_copy = img.copy()
        img_h, img_w = img.shape[:2]
        cv2.line(img, (0, int(img_h / 2)), (img_w, int(img_h / 2)), (0, 0, 200), 1)
        cv2.line(img, (int(img_w / 2), 0), (int(img_w / 2), img_h), (0, 0, 200), 1)

        frame_resize = cv2.resize(img_copy, size, interpolation=cv2.INTER_NEAREST)
        frame_gb = cv2.GaussianBlur(frame_resize, (11, 11), 11)
        # If it is detected with a aera recognized object, the area will be detected ubtil there is no object
        if get_roi and start_pick_up:
            get_roi = False
            frame_gb = getMaskROI(frame_gb, roi, size)

        frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)  # convert the image to LAB space
        return frame_lab, img

    def biggest_area(self, frame_lab):
        '''
        INPUT: Modified image from earlier
        OUTPUT: Largest outline, area, and color
        Calculated largest area from all found contours & color
        '''

        if not self.start_pick_up:
            color_area_max = None
            max_area = 0
            areaMaxContour_max = [0, 0]
            for i in color_range:

                if i in self.target_color:
                    frame_mask = cv2.inRange(frame_lab, color_range[i][0],
                                             color_range[i][1])  # perform bit operations on the original
                    opened = cv2.morphologyEx(frame_mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8))  # open operation
                    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8))  # closed operation
                    contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
                        -2]  # Find the Outline
                    areaMaxContour, area_max = self.get_area_max_contour(contours)  # Find the largest contour
                    if areaMaxContour is not None:
                        if area_max > max_area:  # Find the largest area
                            max_area = area_max
                            color_area_max = i
                            areaMaxContour_max = areaMaxContour
            return areaMaxContour_max, max_area, color_area_max

    def draw_contour(self, img, areaMaxContour_max, max_area, color_area_max):
        if not self.start_pick_up:
            self.rect = cv2.minAreaRect(areaMaxContour_max)
            box = np.int0(cv2.boxPoints(self.rect))
            self.roi = getROI(box)  # Get ROI Area
            self.get_roi = True
            self.img_centerx, self.img_centery = getCenter(self.rect, self.roi, self.size,
                                                           square_length)  # Get the center coordinates of the wood block

            world_x, world_y = convertCoordinate(img_centerx, img_centery,
                                                 self.size)  # Convert to real world Coordinates
            cv2.drawContours(img, [box], -1, range_rgb[color_area_max], 2)
            cv2.putText(img, '(' + str(world_x) + ',' + str(world_y) + ')', (min(box[0, 0], box[2, 0]), box[2, 1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, range_rgb[color_area_max], 1)  # Draw the center point
            return world_x, world_y

    def set_color(self, color_area_max):
        '''
        INPUT: Color string
        OUTPUT: ___
        Converts color string to int
        '''
        if not self.start_pick_up:
            if color_area_max == 'red':  # red max
                color = 1
            elif color_area_max == 'green':  # green max
                color = 2
            elif color_area_max == 'blue':  # blue max
                color = 3
            else:
                color = 0
            self.color_list.append(color)

    def position_confidence(self, world_x, world_y):
        '''
        INPUT: x,y in world coordinates of shape
        OUTPUT: ___
        Sets self.variables depending on how far the center of the shape moved since last check
        Counter counts when shape hasn't moved
        Used to decide when to start grasp
        '''

        distance = math.sqrt(pow(world_x - self.last_x, 2) + pow(world_y - self.last_y,
                                                                 2))  # Compare the last coordinate to determine whether to move
        self.last_x, self.last_y = world_x, world_y
        # Cumulative judgement
        if distance < 0.5:
            self.count += 1
            self.center_list.extend((world_x, world_y))
            if self.start_count_t1:
                self.start_count_t1 = False
                self.t1 = time.time()
            if time.time() - self.t1 > 1:
                rotation_angle = self.rect[2]
                self.start_count_t1 = True
                self.world_X, self.world_Y = np.mean(np.array(self.center_list).reshape(self.count, 2), axis=0)
                self.center_list = []
                self.count = 0
                self.start_pick_up = True
        else:
            self.t1 = time.time()
            self.start_count_t1 = True
            self.center_list = []
            self.count = 0


    def set_colour(self, color_area_max):
        if len(self.color_list) == 3:  # multipe judgments
            # take evaluation value
            color = int(round(np.mean(np.array(self.color_list))))
            color_list = []
            if color == 1:
                self.detect_color = 'red'
                self.draw_color = range_rgb["red"]
            elif color == 2:
                self.detect_color = 'green'
                self.draw_color = range_rgb["green"]
            elif color == 3:
                self.detect_color = 'blue'
                self.draw_color = range_rgb["blue"]
            else:
                self.detect_color = 'None'
                self.draw_color = range_rgb["black"]
        else:
            if not start_pick_up:
                self.draw_color = (0, 0, 0)
                self.detect_color = "None"
        return self.detect_color

class Movement():
    def __init__(self, p):
        self.p = p
        self.stop = False
        self.unreachable = True
        self.rotation_angle = 0
        self.num_cubes = 0
        self.goal = [0, 0]
        self.target = [0, 0]
        self.height = 1.5
        self.orientation = 0

    def stacking_cube(self, goal_coordinate):
        '''
        x,y,z,orientation of each cube
        i take those and do things
        '''
        while True:
            self.num_cubes = len(self.s.cube_pose)
            # how many objects do i have?
            if self.num_cubes == 3:
                dist = np.sqrt((self.s.img_centerx - self.s.cube_pose['red'][0]) ** 2 + (
                            self.s.img_centery - self.s.cube_pose['red'][1]) ** 2)
                if dist < 1.5:
                    # this means red is in the middle and we should move to blue
                    self.height = 4.5
                    self.goal = [self.s.cube_pose['blue'][0], self.s.cube_pose['blue'][1]]
                    self.target = [self.s.cube_pose['red'][0], self.s.cube_pose['red'][1]]
                    self.orientation = self.s.cube_pose['blue'][3]
                    self.rgb('blue')
                else:
                    # red first
                    self.goal = [self.s.cube_pose['red'][0], self.s.cube_pose['red'][1]]
                    self.target = [self.s.img_centerx, self.s.img_centery]
                    self.orientation = self.s.cube_pose['red'][3]
                    self.rgb('red')
            elif self.num_cubes == 2:
                # green last
                self.height = 7.5
                self.goal = self.s.cube_pose['green'][0], self.s.cube_pose['green'][1]
                self.target = [self.s.cube_pose['blue'][0], self.s.cube_pose['blue'][1]]
                self.orientation = self.s.cube_pose['green'][3]
                self.rgb('green')
            elif self.num_cubes == 1:
                # Return to the initial position
                initMove()
                time.sleep(1.5)
            setBuzzer(0.1)
            # find first goal
            result = AK.setPitchRangeMoving((self.goal[0], self.goal[1], 7), -90, -90, 0)
            if result == False:
                self.unreachable = True
            else:
                self.unreachable = False
                time.sleep(result[2] / 1000)  # If you can reach the specified location, get the running time
                # angle for gripper to be rotated to not be in the way
                servo2_angle = getAngle(self.goal[0], self.goal[1], self.orientation)
                Board.setBusServoPulse(1, servo1 - 280, 500)  # Paws Open
                Board.setBusServoPulse(2, servo2_angle, 500)
                time.sleep(0.5)
                # move down to height
                AK.setPitchRangeMoving((self.goal[0], self.goal[1], self.height), -90, -90, 0, 1000)
                time.sleep(1.5)
                # holder closure
                Board.setBusServoPulse(1, servo1, 500)
                time.sleep(0.8)
                # close and lift up
                Board.setBusServoPulse(2, 500, 500)
                AK.setPitchRangeMoving((self.goal[0], self.goal[1], 12), -90, -90, 0, 1000)
                time.sleep(1)
                # move to target position at 12 height
                result = AK.setPitchRangeMoving((self.target[0], self.target[1], 12), -90, -90, 0)
                time.sleep(result[2] / 1000)
                # move to orientation of target
                servo2_angle = getAngle(self.target[0], self.target[1], self.orientation)
                Board.setBusServoPulse(2, servo2_angle, 500)
                time.sleep(0.5)
                # drop to slightly above height
                AK.setPitchRangeMoving((self.target[0], self.target[1], self.height + 3), -90, -90, 0, 500)
                time.sleep(0.5)
                # drop to right height
                AK.setPitchRangeMoving((self.target[0], self.target[1], self.height), -90, -90, 0, 1000)
                time.sleep(0.5)
                # Open the Claws, Put down the object
                Board.setBusServoPulse(1, servo1 - 200, 500)
                time.sleep(0.8)
                # lift back up some
                AK.setPitchRangeMoving((self.target[0], self.target[1], 12), -90, -90, 0, 800)
                time.sleep(0.8)
                # Return to the initial position
                initMove()
                time.sleep(1.5)

    def pickup_cube(self):
        # place coordinates
        coordinate = {
            'red': (-15 + 0.5, 12 - 0.5, 1.5),
            'green': (-15 + 0.5, 6 - 0.5, 1.5),
            'blue': (-15 + 0.5, 0 - 0.5, 1.5),
        }

if __name__ == '__main__':
    initMove()  # Move to starting position
    p = Perception()  # Initialize classes
    m = Movement(p)


    # Start Camera
    my_camera = Camera.Camera()
    my_camera.camera_open()

    while True:
        img = my_camera.frame
        if img is not None:
            frame = img.copy()
            p_frame, Frame = p.preprocess(frame)
            if not s.start_pick_up:
                areaMaxContour_max, max_area, color_area_max = p.biggest_area(p_frame)
            if max_area > 2500:  # Found the largest area
                if not p.start_pick_up:
                    world_x, world_y = s.draw_outline_color(Frame, areaMaxContour_max, max_area, color_area_max)
                if not p.start_pick_up:
                    p.position_confidence(world_x, world_y)
                if not p.start_pick_up:
                    p.set_color(color_area_max)

            # Draw text for seen object
            cv2.putText(Frame, "Color: " + s.detect_color, (10, Frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        s.draw_color, 2)
            cv2.imshow('Frame', Frame)
            set_rgb(s.detect_color)
            key = cv2.waitKey(1)
            if key == 27:
                break
    my_camera.camera_close()
    cv2.destroyAllWindows()
