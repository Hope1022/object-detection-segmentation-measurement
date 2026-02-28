import cv2
import numpy as np
import math
from ultralytics import YOLO

chessboard_size = (9, 6)
square_size = 1.0
min_captures = 15
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ratio = 297/220

objp = np.zeros((chessboard_size[0]*chessboard_size[1],3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)
objp *= square_size

objpoints = []
imgpoints = []
points = []

def draw_circle(event, x, y, flags, params):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) == 2:
            points = []
        points.append((x, y))

cv2.namedWindow("Calibration")
cv2.setMouseCallback("Calibration", draw_circle)

cap = cv2.VideoCapture(0)

address = "http://192.168.8.6:8080/video"
cap.open(address)
captured_count = 0
calibrated = False
camera_matrix = None
dist_coeffs = None

print("\nInstructions: S = capture frame, C = calibrate, Q = quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    display_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret_corners:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(display_frame, chessboard_size, corners2, ret_corners)

    cv2.putText(display_frame, f"Captured: {captured_count}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Calibration", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and ret_corners:
        objpoints.append(objp)
        imgpoints.append(corners2)
        captured_count += 1
        print(f"Captured {captured_count}")

    elif key == ord('c') and captured_count >= min_captures:
        ret_calib, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        print("Calibration Complete!")
        np.save("camera_matrix.npy", camera_matrix)
        np.save("dist_coeffs.npy", dist_coeffs)
        calibrated = True
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if calibrated:
    cap = cv2.VideoCapture(0)
    address = "http://192.168.8.6:8080/video"
    cap.open(address)
    cv2.namedWindow("Undistorted")
    cv2.setMouseCallback("Undistorted", draw_circle)

    model = YOLO("yolov8n-seg.pt")

    coco_classes = [
        "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
        "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
        "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
        "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
        "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
        "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
        "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
        "clock","vase","scissors","teddy bear","hair drier","toothbrush"
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for pt in points:
            cv2.circle(frame, pt, 5, (25,15,255), -1)

        if len(points) == 2:
            pt1 = points[0]
            pt2 = points[1]
            distance_px = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
            distance_cm = distance_px * ratio
            cv2.putText(frame, f"{int(distance_cm)}mm", (pt1[0], pt1[1]-10),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)

        results = model(undistorted)[0]

        for box, cls_id, conf, mask in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf, results.masks.data if results.masks else []):
            x1, y1, x2, y2 = [int(v) for v in box]
            obj_name = coco_classes[int(cls_id)]
            conf_score = float(conf)
            width_px = x2 - x1
            height_px = y2 - y1
            width_cm = width_px * ratio
            height_cm = height_px * ratio
            area_cm2 = width_cm * height_cm

            if results.masks:
                mask_img = (mask.cpu().numpy() * 255).astype(np.uint8)
                mask_img = cv2.resize(mask_img, (undistorted.shape[1], undistorted.shape[0]))
                color = np.random.randint(0, 255, size=3)
                colored_mask = np.zeros_like(undistorted, dtype=np.uint8)
                colored_mask[mask_img>128] = color
                alpha = 0.4
                undistorted = cv2.addWeighted(undistorted, 1, colored_mask, alpha, 0)

            cv2.rectangle(undistorted, (x1,y1), (x2,y2), (0,0,255), 2)

            cv2.putText(undistorted, f"{obj_name} {conf_score:.2f}", (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.putText(undistorted, f"W:{int(width_cm)} H:{int(height_cm)} A:{int(area_cm2)}cm^2",
                (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Undistorted", undistorted)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()