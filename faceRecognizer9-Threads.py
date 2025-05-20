from threading import Thread
import time

def BigBox(color):
    while True:
        print(color,'Big Box is Open')
        time.sleep(5)
        print(color,'Big Box is CLosed')
        time.sleep(5)

def SmallBox(color):
    while True:
        print(color, 'Small Box is Open')
        time.sleep(1)
        print(color, 'Small Box is CLosed')
        time.sleep(1)

bigBoxThread= Thread(target= BigBox, args= ('red',))
smallBoxThread= Thread(target= SmallBox, args=('blue',))

bigBoxThread.daemon=True
smallBoxThread.daemon=True

bigBoxThread.start()
smallBoxThread.start()
while True:
    pass