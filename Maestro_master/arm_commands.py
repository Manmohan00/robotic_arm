
import maestro

def move_arm():
    servo = maestro.Controller('COM6')
    # servo.setAccel(0,4)      #set servo 0 acceleration to 4
    # servo.setTarget(0,3000)  #set servo to move to center position
    # servo.setAccel(1,4)      #set servo 0 acceleration to 4
    # servo.setTarget(1,3000)
    # servo.setAccel(2,4)      #set servo 0 acceleration to 4
    # servo.setTarget(2,3000)
    # servo.setAccel(3,4)      #set servo 0 acceleration to 4
    # servo.setTarget(3,3000)
    # servo.setAccel(4,4)      #set servo 0 acceleration to 4
    # servo.setTarget(4,3000)
    servo.setAccel(2,4)      #set servo 0 acceleration to 4
    servo.setTarget(2,10000)
    # servo.setAccel(2,4)      #set servo 0 acceleration to 4
    # servo.setTarget(2,3000)
    # servo.setAccel(2,4)      #set servo 0 acceleration to 4
    # servo.setTarget(2,3000)
    #servo.setSpeed(1,10)     #set speed of servo 1
    #x = servo.getPosition(1) #get the current position of servo 1
    servo.close()
    
move_arm()    