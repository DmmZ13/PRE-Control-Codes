import cv2

# Substitua pelos valores obtidos no cv2.selectROI
x, y, w, h = 430, 100, 600, 400  # Exemplo de valores

cap = cv2.VideoCapture("/home/ziqi/pre_ws/dataset_steel_red_green/synchronized_dataset_1/zed_right.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Aplica o crop no frame em tempo real: frame[y:y+h, x:x+w]
    cropped = frame[y:y+h, x:x+w]

    cv2.imshow("Preview do Crop", cropped)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# zed robot crop x, y, w, h = 350, 0, 720, 720 
# zed left crop x, y, w, h = 120, 0, 660, 660  
# zed right crop x, y, w, h = 100, 0, 960, 720
# new zed right crop x, y, w, h = 430, 100, 600, 400 