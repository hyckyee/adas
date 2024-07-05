import cv2
import numpy as np
import pyautogui
import time
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import font as tkfont

# Tkinter 초기화
root = tk.Tk()

# 사용할 폰트 설정 (영어 텍스트를 지원하는 폰트를 선택)
font = tkfont.Font(family="Arial", size=12)

# YOLO 모델 파일 경로 설정
weights_path = "yolov3.weights"
cfg_path = "yolov3.cfg"
names_path = "coco.names"

# YOLO 모델 로드
net = cv2.dnn.readNet(weights_path, cfg_path)

# 클래스 로드
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# YOLO 출력층 설정
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 화면 캡처 함수
def capture_screen():
    try:
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    except Exception as e:
        print(f"화면 캡처 중 오류 발생: {e}")
        return None

# 차량 감지 및 거리 계산 함수
def detect_and_calculate_distances(frame):
    try:
        height, width, channels = frame.shape

        # YOLO 입력 준비
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        # 객체 감지
        outs = net.forward(output_layers)

        # 감지된 객체 정보 저장할 리스트 초기화
        class_ids = []
        confidences = []
        boxes = []
        detected_cars = []

        # 감지된 객체 정보 저장
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # 감지된 차량 추가
                    detected_cars.append((x, y, w, h))

        # 검출된 객체 수 확인
        if len(boxes) == 0 or len(confidences) == 0 or len(class_ids) == 0:
            print("감지된 객체 없음.")
            return None, None, None, None

        # Non-Maximum Suppression (NMS) 적용
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        detected_car = None
        detected_front_car = None

        # 차량 탐지 및 거리 계산
        detected_cars = []
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                if label == "car":
                    detected_cars.append((x, y, w, h))

        # 만약 감지된 차량이 2개 이상이면, 두 번째 차량과 앞 차량을 식별
        if len(detected_cars) >= 2:
            detected_cars = sorted(detected_cars, key=lambda x: x[0], reverse=True)  # x 좌표 기준으로 정렬

            car1 = detected_cars[0]  # 맨 뒤의 차 (자기 자신)
            car2 = detected_cars[1]  # 그 앞의 차 (따라갈 차)

            # 두 차의 거리 계산
            distance = calculate_distance(car1, car2)

            detected_car = car1
            detected_front_car = car2

            return distance, detected_car, detected_front_car, boxes

        # 만약 여기까지 왔다면 앞 차량이 탐지되지 않은 경우이므로 None을 반환
        return None, None, None, None

    except Exception as e:
        print(f"차량 감지 및 거리 계산 중 오류 발생: {e}")
        return None, None, None, None

# 거리 계산 함수 (단순 예시)
def calculate_distance(car1, car2):
    try:
        # 차량 크기에 따른 거리 추정
        # 이 부분은 실제 데이터에 기반한 캘리브레이션 필요
        focal_length = 700  # 임의의 초점 거리
        real_car_width = 1.8  # 차량의 실제 너비 (미터)

        x1, y1, w1, h1 = car1
        x2, y2, w2, h2 = car2

        center_x1 = x1 + w1 // 2
        center_y1 = y1 + h1 // 2
        center_x2 = x2 + w2 // 2
        center_y2 = y2 + h2 // 2

        # 두 차의 중심 간 거리 계산
        distance_pixels = np.sqrt((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2)
        distance_meters = (real_car_width * focal_length) / distance_pixels

        return distance_meters

    except Exception as e:
        print(f"거리 계산 중 오류 발생: {e}")
        return None

# 속력 계산 함수 (단순 예시)
def calculate_speed(distance_old, distance_new, time_interval):
    try:
        if distance_old is None or distance_new is None:
            return None
        speed = abs(distance_new - distance_old) / time_interval
        return speed

    except Exception as e:
        print(f"속력 계산 중 오류 발생: {e}")
        return None

# 메인 루프
if __name__ == "__main__":
    try:
        # 초기화
        previous_distance = None
        start_time = time.time()

        while True:
            frame = capture_screen()
            distance, car_box, front_car_box, all_boxes = detect_and_calculate_distances(frame)

            if distance is not None and car_box is not None and front_car_box is not None and all_boxes is not None:
                current_time = time.time()
                elapsed_time = current_time - start_time

                # 앞 차량 주위에 초록색 상자 그리기
                x, y, w, h = front_car_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 맨 뒤의 차량에 빨간색 상자 그리기
                x2, y2, w2, h2 = car_box
                cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

                # 검출된 다른 차량들에 노란색 상자 그리기
                for box in all_boxes:
                    if box != car_box and box != front_car_box:
                        x3, y3, w3, h3 = box
                        cv2.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (0, 255, 255), 2)

                # 차량 사이의 거리 표시
                cv2.putText(frame, f"Distance: {distance:.2f} m", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.line(frame, (car_box[0] + car_box[2] // 2, car_box[1] + car_box[3] // 2),
                         (front_car_box[0] + front_car_box[2] // 2, front_car_box[1] + front_car_box[3] // 2),
                         (255, 0, 0), 2)

                # 검출된 상자를 이용하여 앞 차량의 속력 계산
                front_car_speed = calculate_speed(previous_distance, distance, elapsed_time)
                if front_car_speed is not None:
                    cv2.putText(frame, f"Front Car Speed: {front_car_speed:.2f} m/s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                previous_distance = distance

                # 주석 추가된 이미지를 Matplotlib을 사용하여 표시
                plt.figure(figsize=(10, 8))
                plt.imshow(frame[:, :, ::-1])  # BGR을 RGB로 변환
                plt.title(f"Distance to Front Car: {distance:.2f} meters")
                plt.axis('off')
                plt.show()

            time.sleep(0.1)

    except Exception as ex:
        print(f"오류 발생: {ex}")
