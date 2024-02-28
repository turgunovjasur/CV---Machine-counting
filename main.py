import cv2
import numpy as np
import pandas as pd

cap = cv2.VideoCapture('D:\car\Moshina.mp4')

line_start = (500, 400)
line_end = (500, 500)

total_vehicle_count = 0
line_crossed = False

fgbg = cv2.createBackgroundSubtractorMOG2()

# Excel faylni yaratish
columns = ['Frame', 'Vehicle Count']
data = []
excel_data = pd.DataFrame(data, columns=columns)

frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    fgmask = fgbg.apply(frame)

    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_count = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Line ni kesib o'tgan moshinalarni aks ettirish
        center_x = int((line_start[0] + line_end[0]) / 2)
        cv2.line(frame, (center_x, line_start[1]), (center_x, line_end[1]), (0, 0, 255), 2)

        if y < line_start[1] < y + h and ((x < line_start[0] < x + w) or (x < line_end[0] < x + w)):
            vehicle_count += 1

            if not line_crossed:
                total_vehicle_count += 1
                line_crossed = True
        else:
            line_crossed = False

    # Moshina sanog'ini chiqarish
    cv2.putText(frame, f'Машина саноғи: {vehicle_count}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
    cv2.putText(frame, f'Транспортных средств: {total_vehicle_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)
    cv2.imshow('Video', frame)

    # Excel faylga ma'lumotlar qo'shish
    excel_data = pd.concat(
        [excel_data, pd.DataFrame({'Frame': [frame_number], 'Vehicle Count': [total_vehicle_count]})],
        ignore_index=True)

    if cv2.waitKey(30) & 0xFF == 27:
        break

    frame_number += 1

cap.release()
cv2.destroyAllWindows()

# Excel faylni saqlash
excel_data.to_excel(r'D:\car\vehicle_count_data.xlsx', index=False)
