# coding: utf-8
import cv2
import torch
import ast
import random

torch.nn.Module.dump_patches = True

model = torch.hub.load('./', 'custom', '/home/kaf/PycharmProjects/pythonProject4/RSI-Net/yolov5/penis/best.pt', source='local')

model.conf = 0.008 # Измените значение на меньшее
model.iou = 0.5
model.agnostic = True  # NMS class-agnostic
model.multi_label = False # NMS multiple labels per box
model.max_det = 1  # maximum number of detections per image
model.cuda()


# Попробуйте использовать доступную камеру
cap = cv2.VideoCapture(0)  # Используйте индекс из вывода, который работает
if not cap.isOpened():
    print("Error: Unable to open any camera.")
    exit()

# Ваш остальной код
class_mapping = ['F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16'
                 ,'F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16','F-16']

with torch.no_grad():
    while True:
        # Чтение кадра из видеопотока
        ret, frame = cap.read()

        # Проверка успешности чтения кадра
        if not ret:
            print("Error: Unable to read a frame from the camera.")
            break

        # Изменение размера кадра
        if frame is not None:
            frame = cv2.resize(frame, (640, 480))  # Подстройте разрешение по необходимости

        # Dummy data for icm20948, replace it with your actual data
        icm = [0, 0, 0]

        # Выполнение детекции объектов
        results = model(frame)
        predictions = results.pred[0]
        print("Predictions:", predictions)
        boxes = predictions[:, :4]
        key = cv2.waitKey(27)

        for box in boxes:
            x1, y1, x2, y2 = box.cpu().numpy()
            # cls = predictions[:6]
            # class_name = f"Class: {int(cls)}"
            predictions_tensor = torch.tensor(predictions, device='cuda:0')

            # Преобразовываем тензор в NumPy массив
            predictions_numpy = predictions_tensor.cpu().numpy()

            # Получаем последний обнаруженный объект
            last_prediction = predictions_numpy[-1]

            # Извлекаем предсказанный класс (предположим, что индекс класса - это второй элемент)
            predicted_class = int(last_prediction[5])
            predicted_class_name = class_mapping[predicted_class]

            # Выводим предсказанный класс
            print("Predicted Class:", predicted_class)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_class_name}", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Отображение кадра
        cv2.imshow('Object Detection', frame)

        if key == ord('q'):
            break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

