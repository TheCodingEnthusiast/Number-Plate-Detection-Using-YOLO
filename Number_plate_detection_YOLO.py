import cv2
import torch
import argparse
import numpy as np
from pathlib import Path
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

def detect_number_plate(img, model, device, conf_thres=0.5, iou_thres=0.45):
    # Resize image
    img = cv2.resize(img, (640, 640))
    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize image
    img = img / 255.0
    # Convert to torch tensor
    img = torch.from_numpy(img.transpose(2,0,1)).float().to(device)
    # Reshape tensor
    img = img.unsqueeze(0)
    # Inference
    pred = model(img)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    # If no detection found, return None
    if len(pred) == 0:
        return None
    # Get coordinates of the detection
    bbox = pred[0][:4].cpu().numpy()
    # Scale coordinates back to original image size
    bbox = scale_coords(img.shape[2:], bbox, img.shape[2:]).round()
    # Crop image using the coordinates
    x1, y1, x2, y2 = bbox.astype(np.int32)
    plate = img[0, :, y1:y2, x1:x2]
    # Convert tensor to numpy array
    plate = plate.cpu().numpy().transpose(1,2,0)
    # Convert back to BGR
    plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
    return plate

if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='path to input image')
    args = parser.parse_args()

    # Load model
    weights = 'yolov5s.pt' # path to pre-trained weights
    model = attempt_load(weights, map_location=torch.device('cpu')).autoshape() 

    # Select device
    device = select_device('')

    # Load image
    img = cv2.imread(args.image)

    # Detect number plate
    plate = detect_number_plate(img, model, device)

    if plate is not None:
        # Display image with detected number plate
        cv2.imshow('Number Plate Detection', plate)
        cv2.waitKey(0)
    else:
        print('Number plate not found')
