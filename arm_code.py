from Maestro_master import maestro

def move(finger: int,position: int, SIMULATION_MODE):
    if SIMULATION_MODE:
        print('received command to arm ....')
        return
    
    # servo = maestro.Controller('COM6') 
    servo = maestro.Controller('/dev/ttyACM0')
    speed = 2
    """
    :param finger: 0: Thumb, 1: Index, 2: Middle, 3: Ring, 4: Little
    :param position: 0: Closed, 1: Half Open, 2: Full Open
    :param servo: Maestro Controller
    :return: None
    """
    servo.setAccel(finger, speed)
    
    print('received command to arm ....')
    
    if position == 0:
        servo.setTarget(finger, 3000)
    elif position == 1:
        servo.setTarget(finger, 6000)
    elif position == 2:
        servo.setTarget(finger, 9000)
        
    servo.close()


# servo = maestro.Controller('COM18')


#move(finger=0,position=0)
#move(finger=1,position=0)
#move(finger=2,position=0)
#move(finger=3,position=0)
#move(finger=4,position=0)


#move(finger=2,position=2)
#move(finger=3,position=2)


# servo.close()

