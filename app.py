from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename
import os
import sys
sys.path.append('yolov7')
import cv2
import torch
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
 
#*** Backend operation
 
# WSGI Application
# Provide template folder name
# The default folder name should be "templates" else need to mention custom folder name
 
# Accepted image for to upload for object detection model
UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
 
app = Flask(__name__, template_folder='templates', static_folder='uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'Bla Bla Bla'

classes_to_filter = ['with_mask', 'without_mask']
opt = {
    "weights": "runs/train/exp/weights/best.pt", # Path to weights file default weights are for nano model
    "img-size": 640, # default image size
    "conf-thres": 0.25, # confidence threshold for inference.
    "iou-thres": 0.45, # NMS IoU threshold for inference.
    "device": '0',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": classes_to_filter  # list of classes to filter or None
}

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


# Detect one single image
def detect_single_image(source_image_path):
    with torch.no_grad():
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    img0 = cv2.imread(source_image_path)
    img = letterbox(img0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]

    # Apply NMS
    classes = None
    if opt['classes']:
        classes = []
        for class_name in opt['classes']:
            classes.append(opt['classes'].index(class_name))

    pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=classes, agnostic=False)
    t2 = time_synchronized()
    for i, det in enumerate(pred):
        s = ''
        s += '%gx%g ' % img.shape[2:]  # print string
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        for *xyxy, conf, cls in reversed(det):
            label = f'{names[int(cls)]} {conf:.2f}'
            plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=2)
    return img0


@app.route('/', methods=['GET', 'POST'])
def home():
        return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        
        #input image
        uploaded_img = request.files['uploaded-file']
        img_filename = 'input_image.jpg'
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        
        #predicting section
        source_image_path = full_filename
        predicted_image = detect_single_image(source_image_path)
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.jpg')
        cv2.imwrite(output_image_path, predicted_image)
        
        #upload section
        return render_template('index.html', file_url=full_filename, predicted_url=output_image_path)

# flask clear browser cache (disable cache)
# Solve flask cache images issue
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
 
if __name__=='__main__':
    app.run(debug = True)