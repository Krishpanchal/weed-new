from werkzeug.datastructures import FileStorage
from PIL import Image as im
from flask import Flask, request, jsonify
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.utils import cloudinary_url
from dotenv import load_dotenv
import os
load_dotenv()
app = Flask(__name__)


labelsPath = 'obj.names'
print(labelsPath)
# load weights and cfg
weightsPath = 'crop_weed_detection.weights'
configPath = 'crop_weed.cfg'
LABELS = open(labelsPath).read().strip().split("\n")




def upload_file(file_to_upload):
    file_loc = open('output.png', 'rb')
    file = FileStorage(file_loc)
    # file.save(dst='output1.png')
    # file_loc.close()
    print(f'------------------------------------------------->{type(file)}')
    app.logger.info('in upload route')
    cloudinary.config(cloud_name=os.getenv('CLOUD_NAME'), api_key=os.getenv('CLOUD_API'),
                      api_secret=os.getenv('API_SECRET'))
    upload_result = None
    app.logger.info('%s file_to_upload', file)
    upload_result = cloudinary.uploader.upload(file, folder='ml')
    app.logger.info(upload_result)
    app.logger.info(type(upload_result))
    print(f"----------------------------->{ upload_result['url']}")
    url = upload_result['url']
    print(type(url))
    return url


def predict_weed(path):
    # color selection for drawing bbox
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # load our input image and grab its spatial dimensions
    image = cv2.imread(path)
    (H, W) = image.shape[:2]
    # parameters
    confi = 0.5
    thresh = 0.5
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (512, 512), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confi:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi, thresh)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x+15, y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    det = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = im.fromarray(det)
    data.save('output.png')
    result = upload_file('output.png')
    return result
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


@app.route('/weed', methods=['POST'])
def handle_weed():
    file_to_upload = request.files['file']
    file_to_upload.save('input_file_weed.jpg')
    result = predict_weed('input_file_weed.jpg')
    return jsonify({"result": str(result)}), 200

@app.route('/', methods=['POST'])
def handle():
    return jsonify({'hello': 'world'}), 200


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
