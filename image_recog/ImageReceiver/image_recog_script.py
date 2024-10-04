import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
from ultralytics import YOLO

# connection to rpi
# from Client import PCClient
# pc_client = PCClient(ip="192.168.32.1", port=5000)  # Use the RPi's IP


def load_model(model_path):
    return YOLO(model_path)

def draw_bbox(image, bbox, label, color=(0, 255, 0)):
    xmin, ymin, xmax, ymax = bbox
    image = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
    image = cv2.putText(image, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def predict_image(image_path, model):
    # Load the image
    img = Image.open(image_path)
    img_np = np.array(img)  # Convert image to NumPy array for OpenCV processing
    
    # Predict the image using the model
    results = model.predict(img)
    result = results[0]
    
    # Move tensors to CPU and convert to numpy arrays
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = [model.names[int(cls)] for cls in result.boxes.cls.cpu().numpy()]
    
    # Convert results to a pandas dataframe
    df_results = pd.DataFrame(boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])
    df_confidence = pd.DataFrame(confidences, columns=['confidence'])
    df_name = pd.DataFrame(classes, columns=['name'])
    
    df_results = pd.concat([df_results, df_confidence, df_name], axis=1)
    
    # Calculate the height and width of the bounding box and the area
    df_results['bbox_height'] = df_results['ymax'] - df_results['ymin']
    df_results['bbox_width'] = df_results['xmax'] - df_results['xmin']
    df_results['bbox_area'] = df_results['bbox_height'] * df_results['bbox_width']
    
    # Rank objects by estimated distance (largest bbox area is closest)
    df_results = df_results.sort_values('bbox_area', ascending=False)
    
    # Filter out unwanted labels (e.g., 'Bullseye_id10')
    df_results = df_results[df_results['name'] != 'Bullseye_id10']
    
    # Draw bounding boxes on the image
    for _, row in df_results.iterrows():
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        label = row['name']
        img_np = draw_bbox(img_np, bbox, label)
    
    # Convert annotated image back to PIL Image
    annotated_img = Image.fromarray(img_np)
    
    # Return labels ranked by estimated distance and the annotated image
    labels_ranked_by_distance = df_results['name'].tolist()
    
    return labels_ranked_by_distance, annotated_img

def main(image_path, model_path):
    # Load the model
    model = load_model(model_path)
    
    # Predict and get ranked labels and annotated image
    labels, annotated_image = predict_image(image_path, model)
    
    # Save or display the annotated image
    annotated_image_path = 'annotated_image.jpg'
    annotated_image.save(annotated_image_path)

    # for label in labels:
    #     if label == 'bullseye-id10':
    #         pc_client.send('c')
    #         break
    #     else:
    #         pc_client.send('s')
    #         break
    
    # print(f"Recognised Labels ranked by estimated distance from the camera: {labels}")
    print(f"Recognised image saved as {annotated_image_path}")

if __name__ == "__main__":
    # Update image_path to the rpi image below:
    image_path = '/Users/MtT/Desktop/Image Recog/photo_2024-09-14-13-14-43_jpeg.rf.714d2c6f6509cbe6b8f963af11078a9f.jpg'
    # Update model path upon integration:
    model_path = '/Users/MtT/Desktop/Image Recog/best.pt'
    
    main(image_path, model_path)
