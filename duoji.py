import Jetson.GPIO as GPIO
import time

"""
    jetson nano 控制舵机代码
"""

# 设置GPIO模式
GPIO.setmode(GPIO.BOARD)

# 定义舵机信号引脚
servo_pin = 12

# 设置GPIO引脚为输出
GPIO.setup(servo_pin, GPIO.OUT)

# 创建 PWM 对象
pwm = GPIO.PWM(servo_pin, 50)  # 50 Hz 的 PWM 信号

# 启动 PWM，设置初始占空比（通常是 7.5%）
pwm.start(7.5)

try:
    while True:
        # 移动舵机到一个位置
        pwm.ChangeDutyCycle(2.5)  # 最小位置
        time.sleep(1)
        pwm.ChangeDutyCycle(7.5)  # 中间位置
        time.sleep(1)
        pwm.ChangeDutyCycle(12.5)  # 最大位置
        time.sleep(1)

except KeyboardInterrupt:
    pass

# 清理GPIO
pwm.stop()
GPIO.cleanup()
