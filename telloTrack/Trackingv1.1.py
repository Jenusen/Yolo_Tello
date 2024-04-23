import torch
import cv2
import numpy as np
from djitellopy import Tello
import time
import keyboard
import threading

global x_speed, y_speed, z_speed

class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output


count = 0
# Load YOLOv5 model
model = torch.hub.load('E:/pycharmFile/ObjectTracking1/vision-based-object-detection/yolov5_configurate/ultralytics/yolov5', 'custom', path='E:/pycharmFile/ObjectTracking1/vision-based-object-detection/yolov5_configurate/runs/train/exp27/weights/best.pt',source='local')

# Connect to Tello tello
time.sleep(15)
tello = Tello()
tello.connect()
tello.takeoff()
tello.streamon()

last_command_time = time.time()
COMMAND_INTERVAL = 10  # seconds
OBJECT_THRESHOLD = 120  # Minimum distance to the edge of the frame

pid_x = PID(0.2, 0.005, 0.05)
pid_y = PID(0.2, 0.005, 0.05)
pid_z = PID(0.2, 0.005, 0.05)

MINIMUM_DISTANCE = 50  # Minimum distance to maintain from the object
ANGLE_THRESHOLD = 10  # Minimum angle change to consider a turn
STATIONARY_THRESHOLD = 10  # Maximum movement to consider the object stationary

def calculate_angle(bbox):
    x1, y1, x2, y2 = bbox[:4].astype(int)
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle

prev_angle = None
prev_center_x = None
prev_center_y = None

LOST_OBJECT_TIMEOUT = 5  # Time to wait before starting a search for a lost object
SEARCH_HEIGHT_STEP = 20  # Height to increase for each search step
object_lost_time = None
search_height = 0
searching = False

def initKeyboard():
    keyboard.on_press(onKeyPress)

# 处理按键事件
def onKeyPress(event):
    key = event.name
    if key == 'up':
        tello.takeoff()
    elif key == 'down':
        tello.land()
    elif key == 'j':
        tello.send_rc_control(-70, 0, 0, 0)
    elif key == 'l':
        tello.send_rc_control(70, 0, 0, 0)
    elif key == 'i':
        tello.send_rc_control(0, 70, 0, 0)
    elif key == 'k':
        tello.send_rc_control(0, -70, 0, 0)
    elif key == 'w':
        tello.send_rc_control(0, 0, 70, 0)
    elif key == 's':
        tello.send_rc_control(0, 0, -70, 0)
    elif key == 'a':
        tello.send_rc_control(0, 0, 0, -70)
    elif key == 'd':
        tello.send_rc_control(0, 0, 0, 70)


font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255, 0, 0)
lineThickness = 1

keyboardThread = threading.Thread(target=initKeyboard)
keyboardThread.start()

# Loop through frames from Tello stream
while True:
    start_time = time.time()
    # Get frame from Tello stream
    frame = tello.get_frame_read().frame

    # Run YOLOv5 detection
    results = model(frame)
    bboxes = results.pandas().xyxy[0].values

    # Draw bounding boxes around detected objects
    if len(bboxes) > 0:
        for bbox in bboxes:
            if bbox[5] == 0:
                x1, y1, x2, y2, height = bbox[:4].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Send Tello commands to track IROBOT
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                img_center_x = frame.shape[1] / 2
                img_center_y = frame.shape[0] / 2
                # img_center_z = frame.shape[0] / 2

                x_dist = center_x - img_center_x
                y_dist = center_y - img_center_y
                z_dist = height - frame.shape[0] / 2 # Assuming img_center_z is the image center height

                dt = time.time() - start_time
                x_speed = int(pid_x.update(x_dist, dt))
                y_speed = int(pid_y.update(y_dist, dt))
                z_speed = int(pid_z.update(z_dist, dt))
                x_speed = max(10, min(x_speed, 40))
                y_speed = max(10, min(y_speed, 40))
                z_speed = max(10, min(z_speed, 40))

                tello.send_rc_control(x_speed, y_speed, z_speed)



                if prev_center_x is not None and prev_center_y is not None:
                    x_move = abs(center_x - prev_center_x)
                    y_move = abs(center_y - prev_center_y)

                    if x_move <= STATIONARY_THRESHOLD and y_move <= STATIONARY_THRESHOLD:
                        tello.send_rc_control(0, 0, 0, 0)
                        print("Hovering")
                    else:
                        if abs(x_dist) > OBJECT_THRESHOLD:
                            if x_dist > 0 and x_dist > MINIMUM_DISTANCE:
                                tello.move_right(max(20, x_speed))
                                print(f'Moving right with speed: {x_speed}')
                            elif x_dist < 0 and abs(x_dist) > MINIMUM_DISTANCE:
                                tello.move_left(max(20, x_speed))
                                print(f'Moving left with speed: {x_speed}')

                        if abs(y_dist) > OBJECT_THRESHOLD:
                            if y_dist > 0 and y_dist > MINIMUM_DISTANCE:
                                tello.move_forward(max(20, y_speed))
                                print(f'Moving forward with speed: {y_speed}')
                            elif y_dist < 0 and abs(y_dist) > MINIMUM_DISTANCE:
                                tello.move_back(max(20, y_speed))
                                print(f'Moving back with speed: {y_speed}')

                prev_center_x = center_x
                prev_center_y = center_y

                if prev_angle is not None:
                    angle = calculate_angle(bbox)
                    angle_diff = angle - prev_angle

                    if abs(angle_diff) > ANGLE_THRESHOLD:
                        if angle_diff > 0:
                            tello.rotate_counter_clockwise(int(abs(angle_diff)))
                            print(f'Rotating counterclockwise by {abs(angle_diff)} degrees')
                        else:
                            tello.rotate_clockwise(int(abs(angle_diff)))
                            print(f'Rotating clockwise by {abs(angle_diff)} degrees')

                prev_angle = calculate_angle(bbox)
                object_lost_time = None
                searching = False


# 找不到物体的跟踪逻辑
# 这里可以改为切换成单控制模式
    else:
        if object_lost_time is None:
            object_lost_time = time.time()
        elif time.time() - object_lost_time > LOST_OBJECT_TIMEOUT:
            if count<2:
                if not searching:
                    search_height += SEARCH_HEIGHT_STEP
                    tello.move_up(search_height)
                    searching = True
                else:
                    tello.rotate_counter_clockwise(360)
                    searching = False
                    object_lost_time = time.time()
                    count += 1
            else:
                initKeyboard()
                onKeyPress()
                # tello.land()
                # tello.end()

    # Display frame
    cv2.imshow('Tello Stream', frame)

    InfoText = "battery : {0}% height: {1}cm   time: {2}".format(tello.get_battery(), tello.get_height(),
                                                                 time.strftime("%H:%M:%S",
                                                                               time.localtime()))
    cv2.putText(frame, InfoText, (10, 20), font, fontScale, (0, 0, 255), lineThickness)
    # if key == 1:
    InfoText = "Command : x_speed:{0}% y_speed:{1} z_speed:{2}".format(x_speed, y_speed, z_speed)
    cv2.putText(frame, InfoText, (10, 40), font, fontScale, (0, 0, 255), lineThickness)

    tello.send_rc_control(x_speed, y_speed, z_speed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if time.time() - last_command_time > COMMAND_INTERVAL:
        tello.send_control_command('command')
        last_command_time = time.time()

# Clean up
cv2.destroyAllWindows()

tello.streamoff()
tello.land()
tello.end()



