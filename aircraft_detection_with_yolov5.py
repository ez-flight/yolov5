import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Загрузка предварительно обученной модели
weights_path = '/home/kaf/PycharmProjects/pythonProject4/RSI-Net/yolov5/penis/best.pt'  # Замените на фактический путь к вашему файлу весов
model = attempt_load(weights_path).float()
model.eval()

# Инициализация видеопотока
cap = cv2.VideoCapture(1)  # Или указать путь к видеофайлу

while True:
    # Считывание кадра из видеопотока
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в тензор
    input_tensor = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2).float()  # Перестановка осей
    # Передача изображения модели
    with torch.no_grad():
        output = model(input_tensor)

    # Применение non-maximum suppression для получения детекций
    pred = non_max_suppression(output[0])

    # Отображение результата
    # Отображение результата
    if pred[0] is not None:
        for det in pred[0]:
            x_min, y_min, x_max, y_max, conf, cls = det
            # Масштабирование координат относительно размера входного изображения
            x_min, y_min, x_max, y_max = x_min * frame.shape[1], y_min * frame.shape[0], x_max * frame.shape[1], y_max * \
                                         frame.shape[0]

            class_name = f"Class: {int(cls)}"
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}", (int(x_min), int(y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Отображение результата
    cv2.imshow('Inference Result', frame)

    # Для выхода из цикла нажмите клавишу 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие окон
cap.release()
cv2.destroyAllWindows()
