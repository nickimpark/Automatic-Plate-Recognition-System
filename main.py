import cv2
import numpy as np
import pytesseract
import pandas as pd
import re


class PlateDetector:
    def __init__(self):
        self.psm = 8 # page segmentation mode
        self.white_list = 'ABCEHKMOPTXY0123456789' # символы для распознавания (в составе автомобильных номеров)
        self.video_source = 'http://raspberrypi.local:8000/stream.mjpg'  # 10 FPS, 640x480
        self.carplate_haar_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_russian_plate_number.xml') # каскад Хаара
        self.plates = list(pd.read_csv('./data/plates.csv')) # база номеров

    # детекция номерной рамки
    def carplate_detect(self, image):
        carplate_overlay = image.copy()
        carplate_rects = self.carplate_haar_cascade.detectMultiScale(carplate_overlay, scaleFactor=1.1, minNeighbors=20)
        if len(carplate_rects) > 0:
            for x, y, w, h in carplate_rects:
                cv2.rectangle(carplate_overlay, (x, y), (x + w, y + h), (0, 255, 0), 5)
                return carplate_overlay, x, y, w, h
        else:
            return carplate_overlay, 0, 0, 0, 0

    # экстракция номера
    def carplate_extract(self, image):

        carplate_rects = self.carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=10)

        for x, y, w, h in carplate_rects:
            carplate_img = image[y + 15:y + h - 10, x + 15:x + w - 20]
            return carplate_img

    # расстояние Левенштейна
    def levenstein(self, str_1, str_2):
        n, m = len(str_1), len(str_2)
        if n > m:
            str_1, str_2 = str_2, str_1
            n, m = m, n

        current_row = range(n + 1)
        for i in range(1, m + 1):
            previous_row, current_row = current_row, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if str_1[j - 1] != str_2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        return current_row[n]

    # функционирование системы
    def operate(self):
        cap = cv2.VideoCapture(self.video_source)
        # press Q or Esc to stop
        while cv2.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            if not hasFrame:
                cv2.waitKey(3000)
                # Release device
                cap.release()
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            carplate_detected, x, y, w, h = self.carplate_detect(frame)
            if x > 0:
                carplate_extracted = self.carplate_extract(carplate_detected)
                carplate_extracted = cv2.cvtColor(carplate_extracted, cv2.COLOR_RGB2GRAY)
                carplate_extracted = cv2.medianBlur(carplate_extracted, 3)
                carplate_extracted = cv2.addWeighted(carplate_extracted,
                                                                     1.5, carplate_extracted, 0, 1.5)
                num_detected = pytesseract.image_to_string(carplate_extracted,
                                                           config=f'--psm {self.psm} --oem 3 -c tessedit_char_whitelist={self.white_list}')
                num_detected = re.sub('[\W_]+', '', num_detected)
                min_levenstein = np.inf
                closest_plate = None
                for plate in self.plates:
                    curr_levenstein = self.levenstein(num_detected, plate)
                    if curr_levenstein < min_levenstein:
                        min_levenstein = curr_levenstein
                        closest_plate = plate
                if (closest_plate is not None) and (min_levenstein <= 2):
                    num_detected = closest_plate
                if num_detected in self.plates:
                    carplate_detected = cv2.putText(carplate_detected, num_detected + ' - welcome',
                                                            (x + w + 20, y + int(0.5 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                            (0, 255, 0), 2, cv2.LINE_AA, False)
                else:
                    cv2.rectangle(carplate_detected, (x, y), (x + w, y + h), (0, 0, 255), 5)
                    carplate_detected = cv2.putText(carplate_detected, num_detected + ' - not registered',
                                                    (x + w + 10, y + int(0.5 * h)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                    (0, 0, 255), 2, cv2.LINE_AA, False)
            carplate_detected = cv2.resize(carplate_detected, (720, 540), interpolation=cv2.INTER_AREA)
            cv2.imshow('plate_detector', carplate_detected)


if __name__ == '__main__':
    pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe' # путь до .exe Tesseract
    detector = PlateDetector()
    detector.operate()
