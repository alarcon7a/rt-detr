import cv2
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

# Cargar el modelo y el procesador de imágenes
image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")

# Mover el modelo a la GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Inicializar la cámara
cap = cv2.VideoCapture(1)

# Establecer la resolución deseada (reducida)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Capturar frame por frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convertir el frame de OpenCV a formato PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Procesar la imagen
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    # Realizar la detección
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-procesar los resultados
    results = image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]], device=device), threshold=0.5)

    # Dibujar los resultados en el frame
    for result in results:
        for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
            box = [int(i) for i in box.tolist()]
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label = f"{model.config.id2label[label_id.item()]}: {score.item():.2f}"
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el resultado
    cv2.imshow('Object Detection', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
