import time
from math import tan, pi
import numpy as np
import cv2

try :
    from ezblock import __reset_mcu__
    from ezblock import *
    
    __reset_mcu__()
    time.sleep(0.01)
except ImportError :
    print ("Simulator")
    from sim_ezblock import *

class Sensors:
    def __init__(self):
        self.S0 = ADC('A0')
        self.S1 = ADC('A1')
        self.S2 = ADC('A2')
        self.trig = Pin('D8')
        self.echo = Pin('D9')

    
    def get_adc_value(self):
        adc_value_list = []
        adc_value_list.append(self.S0.read())
        adc_value_list.append(self.S1.read())
        adc_value_list.append(self.S2.read())
        logging.info("adcs: {0}".format(adc_value_list))
        return adc_value_list  

    def get_distance(self):
        timeout=0.05
    
        self.trig.low()
        time.sleep(0.01)
        self.trig.high()
        time.sleep(0.000015)
        self.trig.low()
        pulse_end = 0
        pulse_start = 0
        timeout_start = time.time()
        # logging.info("echo: {0}".format(self.echo.value()))

        while self.echo.value()==0:
            pulse_start = time.time()
            if pulse_start - timeout_start > timeout:
                return -1
        while self.echo.value()==1:
            pulse_end = time.time()
            if pulse_end - timeout_start > timeout:
                # logging.info("pulse_end: {0} timeout_start: {1}".format(pulse_end,timeout_start))
                return -2
        during = pulse_end - pulse_start
        cm = round(during * 340 / 2 * 100, 2)
        logging.info("distance(cm): {0}".format(cm))
        return cm
        
class Interpreters:
    def __init__(self):
        self.sensitivity = 200
        self.polarity = 1 # Means black line

    
    def get_grayscale_value(self, adcs):
        if abs(adcs[0] - adcs[2]) > self.sensitivity:
            if adcs[0] < adcs[2]:
                if adcs[0] + abs((adcs[2]-adcs[0])/4) > adcs[1]:
                    rob_pos = .5 * self.polarity   
                else:
                    rob_pos = 1* self.polarity
            else:
                if adcs[2]+abs((adcs[2]-adcs[0])/4) < adcs[1]:
                    rob_pos = -1 * self.polarity   
                else:
                    rob_pos = -.5* self.polarity
        else:
            rob_pos = 0
                
        logging.info("robot pos: {0}".format(rob_pos))
        return rob_pos
          

class Controllers:
    
    def __init__(self,m):
       self.line_steering = -30 
       self.m = m
    def line_following(self, rob_pos, speed):
        logging.info("steering angle: {0}, speed: {1}".format(rob_pos*self.line_steering,speed))
        self.m.set_dir_servo_angle(rob_pos*self.line_steering)
        self.m.forward(speed)
        return rob_pos*self.line_steering
    
    def wall_checking(self, cm):
        if 0 < cm < 5:
            logging.info("About to hit an obstacle @ {0}".format(cm))
            self.m.forward(0)
            
            
            
            

if __name__ == "__main__":
    pass 
