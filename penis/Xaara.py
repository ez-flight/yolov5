import cv2

# Загрузка каскада Хаара для распознавания лица
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# Загрузка изображения или видеопотока
# Замените 'input.jpg' на путь к вашему изображению или 0 для использования камеры
input_source = 1  # Можно заменить на 0 для использования камеры

# Инициализация видеопотока
cap = cv2.VideoCapture(input_source)

while True:
    # Считывание кадра из видеопотока
    ret, frame = cap.read()
    if not ret:
        break

    # Преобразование кадра в оттенки серого (улучшает производительность)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.04, minNeighbors=23, minSize=(100, 100))

    # Рисование контуров лиц и добавление подписей
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Интерполяция для увеличения размера кадра без потери качества
    frame = cv2.resize(frame, (2560, 1440), interpolation=cv2.INTER_LINEAR)

    # Отображение результата
    cv2.imshow('Face Detection', frame)

    # Для выхода из цикла нажмите клавишу 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие окон
cap.release()
cv2.destroyAllWindows()
