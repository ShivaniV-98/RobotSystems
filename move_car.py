from picarx_improved import picarx_improved
from time import sleep
class picar_driver(picarx_improved):
    def __init__(self):
        super().__init__()
        self.heading_angle = 0
        self.speed = 0
        return

    def move_forward(self):
        for i in range(10):
            self.forward(self.speed, self.heading_angle)
        self.stop()

    def move_backward(self):
        for i in range(100):
            self.set_dir_servo_angle(0)
        for i in range(10):
            self.backward(self.speed)
        self.stop()

    def move_left(self):
        for i in range(100):
            self.set_dir_servo_angle(self.heading_angle)
        for i in range(10):
            self.forward(self.speed, self.heading_angle)
        self.stop()

    def move_right(self):
        for i in range(100):
            self.set_dir_servo_angle(self.heading_angle)
        for i in range(10):
            self.forward(self.speed, self.heading_angle)
        self.stop()

    def parallel_parking(self, direction='l'):
        if direction == 'l':
            dir_i = -1
        else:
            dir_i = 1
        for i in range(15):
            self.set_dir_servo_angle(30*dir_i)
        for i in range(10):
            self.forward(40)
        for i in range(30):
            self.set_dir_servo_angle(-30*dir_i)
        for i in range(10):
            self.forward(10, -30*dir_i)
        self.stop()

    def k_turning(self, speed,direction = -1):
        self.set_dir_servo_angle(direction * 40)
        picarx_improved.forward(speed, -direction * 40)
        time.sleep(length * .50)
        self.set_dir_servo_angle(-direction * 40)
        self.backward(speed, direction * 40)
        time.sleep(length * .50)
        self.set_dir_servo_angle(direction * 40)
        self.forward(speed, -direction * 40)
        time.sleep(length * .50)

    def keyboard_input(self):
        speed =0
        heading_angle =0
        input_choice = input("used 'w-s-a-d' for 'forward-backward-left-right' maneuvering. 'e' to exit maneuvering, 'p' to park and 'k' for k_turn")

        while input_choice != 'e':
            if input_choice == 'w':
                self.speed = 40
                self.heading_angle = 0
                self.move_forward()
            elif input_choice == 's':
                self.speed = 80
                self.move_backward()
            elif input_choice == 'a':
                self.speed = 40
                self.heading_angle = -30
                self.move_left()
            elif input_choice == 'd':
                self.speed = 40
                self.heading_angle = 30
                self.move_right()
            elif input_choice == 'p':
                get_dir = input("'r' for right park and 'l' for left park")
                self.parallel_parking(get_dir)
            elif input_choice == 'k':
                get_dir = input("'r' for right turn and 'l' for left turn")
                self.k_turning(get_dir)

            input_choice = input("Next Action please : ")


if __name__ == "__main__":
    driver = picar_driver()
    driver.keyboard_input()
