class Pin(object):
    #OUT = GPIO.OUT
    #IN = GPIO.IN
   # IRQ_FALLING = GPIO.FALLING
   #  IRQ_RISING = GPIO.RISING
   #  IRQ_RISING_FALLING = GPIO.BOTH
   #  PULL_UP = GPIO.PUD_UP
   #  PULL_DOWN = GPIO.PUD_DOWN
   #  PULL_NONE = None

    _dict = {
        "BOARD_TYPE": 12,
    }

    _dict_1 = {
        "D0": 17,
        "D1": 18,
        "D2": 27,
        "D3": 22,
        "D4": 23,
        "D5": 24,
        "D6": 25,
        "D7": 4,
        "D8": 5,
        "D9": 6,
        "D10": 12,
        "D11": 13,
        "D12": 19,
        "D13": 16,
        "D14": 26,
        "D15": 20,
        "D16": 21,
        "SW": 19,
        "LED": 26,
        "BOARD_TYPE": 12,
        "RST": 16,
        "BLEINT": 13,
        "BLERST": 20,
        "MCURST": 21,
    }

    _dict_2 = {
        "D0": 17,
        "D1": 4,  # Changed
        "D2": 27,
        "D3": 22,
        "D4": 23,
        "D5": 24,
        "D6": 25,  # Removed
        "D7": 4,  # Removed
        "D8": 5,  # Removed
        "D9": 6,
        "D10": 12,
        "D11": 13,
        "D12": 19,
        "D13": 16,
        "D14": 26,
        "D15": 20,
        "D16": 21,
        "SW": 25,  # Changed
        "LED": 26,
        "BOARD_TYPE": 12,
        "RST": 16,
        "BLEINT": 13,
        "BLERST": 20,
        "MCURST": 5,  # Changed
    }

    def __init__(self, *value):
        # super().__init__()
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setwarnings(False)
        #
        # self.check_board_type()
        #
        # if len(value) > 0:
        #     pin = value[0]
        # if len(value) > 1:
        #     mode = value[1]
        # else:
        #     mode = None
        # if len(value) > 2:
        #     setup = value[2]
        # else:
        #     setup = None
        # if isinstance(pin, str):
        #     try:
        #         self._board_name = pin
        #         self._pin = self.dict()[pin]
        #     except Exception as e:
        #         print(e)
        #         self._error('Pin should be in %s, not %s' % (self._dict.keys(), pin))
        # elif isinstance(pin, int):
        #     self._pin = pin
        # else:
        #     self._error('Pin should be in %s, not %s' % (self._dict.keys(), pin))
        #self._value = 0
        #self.init(mode, pull=setup)
        # self._info("Pin init finished.")
        pass
    def check_board_type(self):
        type_pin = self.dict()["BOARD_TYPE"]
        GPIO.setup(type_pin, GPIO.IN)
        if GPIO.input(type_pin) == 0:
            self._dict = self._dict_1
        else:
            self._dict = self._dict_2

    # def init(self, mode, pull=PULL_NONE):
    #     self._pull = pull
    #     self._mode = mode
    #     if mode != None:
    #         if pull != None:
    #             GPIO.setup(self._pin, mode, pull_up_down=pull)
    #         else:
    #             GPIO.setup(self._pin, mode)

    def dict(self, *_dict):
        if len(_dict) == 0:
            return self._dict
        else:
            if isinstance(_dict, dict):
                self._dict = _dict
            else:
                self._error(
                    'argument should be a pin dictionary like {"my pin": ezblock.Pin.cpu.GPIO17}, not %s' % _dict)

    def __call__(self, value):
        return self.value(value)

    def value(self, *value):
        # if len(value) == 0:
        #     self.mode(self.IN)
        #     result = GPIO.input(self._pin)
        #     # self._debug("read pin %s: %s" % (self._pin, result))
        #     return result
        # else:
        #     value = value[0]
        #     self.mode(self.OUT)
        #     GPIO.output(self._pin, value)
        #     return value
        pass

    def on(self):
        return self.value(1)

    def off(self):
        return self.value(0)

    def high(self):
        return self.on()

    def low(self):
        return self.off()

    def mode(self, *value):
        if len(value) == 0:
            return self._mode
        else:
            mode = value[0]
            self._mode = mode
            GPIO.setup(self._pin, mode)

    def pull(self, *value):
        return self._pull

    def irq(self, handler=None, trigger=None, bouncetime=200):
        self.mode(self.IN)
        GPIO.add_event_detect(self._pin, trigger, callback=handler, bouncetime=bouncetime)

    def name(self):
        return "GPIO%s" % self._pin

    def names(self):
        return [self.name, self._board_name]

    class cpu(object):
        GPIO17 = 17
        GPIO18 = 18
        GPIO27 = 27
        GPIO22 = 22
        GPIO23 = 23
        GPIO24 = 24
        GPIO25 = 25
        GPIO26 = 26
        GPIO4 = 4
        GPIO5 = 5
        GPIO6 = 6
        GPIO12 = 12
        GPIO13 = 13
        GPIO19 = 19
        GPIO16 = 16
        GPIO26 = 26
        GPIO20 = 20
        GPIO21 = 21

        def __init__(self):
            pass


class Servo(object):
    MAX_PW = 2500
    MIN_PW = 500
    _freq = 50

    def __init__(self, pwm):
        # super().__init__()
        # self.pwm = pwm
        # self.pwm.period(4095)
        # #prescaler = int(float(self.pwm.CLOCK) / self.pwm._freq / self.pwm.period())
        # self.pwm.prescaler(prescaler)
        # # self.angle(90)
        pass

    def map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    # angle ranges -90 to 90 degrees
    def angle(self, angle):
        if not (isinstance(angle, int) or isinstance(angle, float)):
            raise ValueError("Angle value should be int or float value, not %s" % type(angle))
        if angle < -90:
            angle = -90
        if angle > 90:
            angle = 90
        High_level_time = self.map(angle, -90, 90, self.MIN_PW, self.MAX_PW)
        # self._debug("High_level_time: %f" % High_level_time)
        pwr = High_level_time / 20000
        # self._debug("pulse width rate: %f" % pwr)
        #value = int(pwr * self.pwm.period())
        # self._debug("pulse width value: %d" % value)
        self.pwm.pulse_width(value)
class I2C(object):
    MASTER = 0
    SLAVE = 1
    RETRY = 5

    def __init__(self, *args, **kargs):  # *args表示位置参数（形式参数），可无，； **kargs表示默认值参数，可无。
        super().__init__()
        self._bus = 1
        #self._smbus = SMBus(self._bus)

    def _i2c_write_byte(self, addr, data):  # i2C 写系列函数
        # self._debug("_i2c_write_byte: [0x{:02X}] [0x{:02X}]".format(addr, data))
        return self._smbus.write_byte(addr, data)

    def _i2c_write_byte_data(self, addr, reg, data):
        # self._debug("_i2c_write_byte_data: [0x{:02X}] [0x{:02X}] [0x{:02X}]".format(addr, reg, data))
        return self._smbus.write_byte_data(addr, reg, data)

    def _i2c_write_word_data(self, addr, reg, data):
        # self._debug("_i2c_write_word_data: [0x{:02X}] [0x{:02X}] [0x{:04X}]".format(addr, reg, data))
        return self._smbus.write_word_data(addr, reg, data)

    def _i2c_write_i2c_block_data(self, addr, reg, data):
        # self._debug("_i2c_write_i2c_block_data: [0x{:02X}] [0x{:02X}] {}".format(addr, reg, data))
        return self._smbus.write_i2c_block_data(addr, reg, data)

    def _i2c_read_byte(self, addr):  # i2C 读系列函数
        # self._debug("_i2c_read_byte: [0x{:02X}]".format(addr))
        return self._smbus.read_byte(addr)

    def _i2c_read_i2c_block_data(self, addr, reg, num):
        # self._debug("_i2c_read_i2c_block_data: [0x{:02X}] [0x{:02X}] [{}]".format(addr, reg, num))
        return self._smbus.read_i2c_block_data(addr, reg, num)

    def is_ready(self, addr):
        addresses = self.scan()
        if addr in addresses:
            return True
        else:
            return False

    def scan(self):  # 查看有哪些i2c设备
        cmd = "i2cdetect -y %s" % self._bus
        _, output = self.run_command(cmd)  # 调用basic中的方法，在linux中运行cmd指令，并返回运行后的内容

        outputs = output.split('\n')[1:]  # 以回车符为分隔符，分割第二行之后的所有行
        # self._debug("outputs")
        addresses = []
        for tmp_addresses in outputs:
            if tmp_addresses == "":
                continue
            tmp_addresses = tmp_addresses.split(':')[1]
            tmp_addresses = tmp_addresses.strip().split(' ')  # strip函数是删除字符串两端的字符，split函数是分隔符
            for address in tmp_addresses:
                if address != '--':
                    addresses.append(int(address, 16))
        # self._debug("Conneceted i2c device: %s"%addresses)                   # append以列表的方式添加address到addresses中
        return addresses

    def send(self, send, addr, timeout=0):  # 发送数据，addr为从机地址，send为数据
        if isinstance(send, bytearray):
            data_all = list(send)
        elif isinstance(send, int):
            data_all = []
            d = "{:X}".format(send)
            d = "{}{}".format("0" if len(d) % 2 == 1 else "",
                              d)  # format是将()中的内容对应填入{}中，（）中的第一个参数是一个三目运算符，if条件成立则为“0”，不成立则为“”(空的意思)，第二个参数是d，此行代码意思为，当字符串为奇数位时，在字符串最强面添加‘0’，否则，不添加， 方便以下函数的应用
            # print(d)
            for i in range(len(d) - 2, -1, -2):  # 从字符串最后开始取，每次取2位
                tmp = int(d[i:i + 2], 16)  # 将两位字符转化为16进制
                # print(tmp)
                data_all.append(tmp)  # 添加到data_all数组中
            data_all.reverse()
        elif isinstance(send, list):
            data_all = send
        else:
            raise ValueError("send data must be int, list, or bytearray, not {}".format(type(send)))

        if len(data_all) == 1:  # 如果data_all只有一组数
            data = data_all[0]
            self._i2c_write_byte(addr, data)
        elif len(data_all) == 2:  # 如果data_all只有两组数
            reg = data_all[0]
            data = data_all[1]
            self._i2c_write_byte_data(addr, reg, data)
        elif len(data_all) == 3:  # 如果data_all只有三组数
            reg = data_all[0]
            data = (data_all[2] << 8) + data_all[1]
            self._i2c_write_word_data(addr, reg, data)
        else:
            reg = data_all[0]
            data = list(data_all[1:])
            self._i2c_write_i2c_block_data(addr, reg, data)

    def recv(self, recv, addr=0x00, timeout=0):  # 接收数据
        if isinstance(recv, int):  # 将recv转化为二进制数
            result = bytearray(recv)
        elif isinstance(recv, bytearray):
            result = recv
        else:
            return False
        for i in range(len(result)):
            result[i] = self._i2c_read_byte(addr)
        return result

    def mem_write(self, data, addr, memaddr, timeout=5000, addr_size=8):  # memaddr match to chn
        if isinstance(data, bytearray):
            data_all = list(data)
        elif isinstance(data, list):
            data_all = data
        elif isinstance(data, int):
            data_all = []
            data = "%x" % data
            if len(data) % 2 == 1:
                data = "0" + data
            # print(data)
            for i in range(0, len(data), 2):
                # print(data[i:i+2])
                data_all.append(int(data[i:i + 2], 16))
        else:
            raise ValueError("memery write require arguement of bytearray, list, int less than 0xFF")
        # print(data_all)
        self._i2c_write_i2c_block_data(addr, memaddr, data_all)

    def mem_read(self, data, addr, memaddr, timeout=5000, addr_size=8):  # 读取数据
        if isinstance(data, int):
            num = data
        elif isinstance(data, bytearray):
            num = len(data)
        else:
            return False
        result = bytearray(self._i2c_read_i2c_block_data(addr, memaddr, num))
        return result

    def readfrom_mem_into(self, addr, memaddr, buf):
        buf = self.mem_read(len(buf), addr, memaddr)
        return buf

    def writeto_mem(self, addr, memaddr, data):
        self.mem_write(data, addr, memaddr)


class PWM(I2C):
    REG_CHN = 0x20
    REG_FRE = 0x30
    REG_PSC = 0x40
    REG_ARR = 0x44

    ADDR = 0x14

    CLOCK = 72000000

    def __init__(self, channel, debug="critical"):
        # super().__init__()
        # if isinstance(channel, str):
        #     if channel.startswith("P"):
        #         channel = int(channel[1:])
        #     else:
        #         raise ValueError("PWM channel should be between [P1, P14], not {0}".format(channel))
        # try:
        #     self.send(0x2C, self.ADDR)
        #     self.send(0, self.ADDR)
        #     self.send(0, self.ADDR)
        # except IOError:
        #     self.ADDR = 0x15
        #
        # self.debug = debug
        # # self._debug("PWM address: {:02X}".format(self.ADDR))
        # self.channel = channel
        # self.timer = int(channel/4)
        # #self.bus = smbus.SMBus(1)
        # self._pulse_width = 0
        # self._freq = 50
        # self.freq(50)
        pass
    def i2c_write(self, reg, value):
        # value_h = value >> 8
        # value_l = value & 0xff
        # # self._debug("i2c write: [0x%02X, 0x%02X, 0x%02X, 0x%02X]"%(self.ADDR, reg, value_h, value_l))
        # self.send([reg, value_h, value_l], self.ADDR)
        pass
    def freq(self, *freq):
        # if len(freq) == 0:
        #     return self._freq
        # else:
        #     self._freq = int(freq[0])
        #     # [prescaler,arr] list
        #     result_ap = []
        #     # accuracy list
        #     result_acy = []
        #     # middle value for equal arr prescaler
        #     st = int(math.sqrt(self.CLOCK/self._freq))
        #     # get -5 value as start
        #     st -= 5
        #     # prevent negetive value
        #     if st <= 0:
        #         st = 1
        #     for psc in range(st,st+10):
        #         arr = int(self.CLOCK/self._freq/psc)
        #         result_ap.append([psc, arr])
        #         result_acy.append(abs(self._freq-self.CLOCK/psc/arr))
        #     i = result_acy.index(min(result_acy))
        #     psc = result_ap[i][0]
        #     arr = result_ap[i][1]
        #     # self._debug("prescaler: %s, period: %s"%(psc, arr))
        #     self.prescaler(psc)
        #     self.period(arr)
        pass

    def prescaler(self, *prescaler):
        # if len(prescaler) == 0:
        #     return self._prescaler
        # else:
        #     self._prescaler = int(prescaler[0]) - 1
        #     reg = self.REG_PSC + self.timer
        #     # self._debug("Set prescaler to: %s"%self._prescaler)
        #     self.i2c_write(reg, self._prescaler)
        pass
    def period(self, *arr):
        # global timer
        # if len(arr) == 0:
        #     return timer[self.timer]["arr"]
        # else:
        #     timer[self.timer]["arr"] = int(arr[0]) - 1
        #     reg = self.REG_ARR + self.timer
        #     # self._debug("Set arr to: %s"%timer[self.timer]["arr"])
        #     self.i2c_write(reg, timer[self.timer]["arr"])
        pass
    def pulse_width(self, *pulse_width):
        # if len(pulse_width) == 0:
        #     return self._pulse_width
        # else:
        #     self._pulse_width = int(pulse_width[0])
        #     reg = self.REG_CHN + self.channel
        #     self.i2c_write(reg, self._pulse_width)
        pass
    def pulse_width_percent(self, *pulse_width_percent):
        # global timer
        # if len(pulse_width_percent) == 0:
        #     return self._pulse_width_percent
        # else:
        #     self._pulse_width_percent = pulse_width_percent[0]
        #     temp = self._pulse_width_percent / 100.0
        #     # print(temp)
        #     pulse_width = temp * timer[self.timer]["arr"]
        #     self.pulse_width(pulse_width)
        pass


class ADC(I2C):
    ADDR = 0x14  # 扩展板的地址为0x14

    def __init__(self, chn):  # 参数，通道数，树莓派扩展板上有8个adc通道分别为"A0, A1, A2, A3, A4, A5, A6, A7"
        super().__init__()
        if isinstance(chn, str):
            if chn.startswith("A"):  # 判断穿境来的参数是否为A开头，如果是，取A后面的数字出来
                chn = int(chn[1:])
            else:
                raise ValueError("ADC channel should be between [A0, A7], not {0}".format(chn))
        if chn < 0 or chn > 7:  # 判断取出来的数字是否在0~7的范围内
            self._error('Incorrect channel range')
        chn = 7 - chn
        self.chn = chn | 0x10  # 给从机地址
        self.reg = 0x40 + self.chn
        # self.bus = smbus.SMBus(1)

    def read(self):  # adc通道读取数---写一次数据，读取两次数据 （读取的数据范围是0~4095）
        # self._debug("Write 0x%02X to 0x%02X"%(self.chn, self.ADDR))
        # self.bus.write_byte(self.ADDR, self.chn)      # 写入数据
        self.send([self.chn, 0, 0], self.ADDR)

        # self._debug("Read from 0x%02X"%(self.ADDR))
        # value_h = self.bus.read_byte(self.ADDR)
        value_h = self.recv(1, self.ADDR)[0]  # 读取数据

        # self._debug("Read from 0x%02X"%(self.ADDR))
        # value_l = self.bus.read_byte(self.ADDR)
        value_l = self.recv(1, self.ADDR)[0]  # 读取数据（读两次）

        value = (value_h << 8) + value_l
        # self._debug("Read value: %s"%value)
        return value

    def read_voltage(self):  # 将读取的数据转化为电压值（0~3.3V）
        return self.read * 3.3 / 4095





