import argparse
import cv2

from yolo import YOLO

ap = argparse.ArgumentParser()
ap.add_argument('-n', '--network', default="normal", help='Network Type: normal / tiny / prn')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=256, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.2, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("starting webcam...")
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

verbose = False

while rval:
    scale_percent = 50 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    if verbose:
        print(f'{width} x {height}')
    dim = (width, height)    
    xframe = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    sframe = cv2.flip(xframe, 1)
    width, height, inference_time, results = yolo.inference(sframe)
    # bbox = yolo.infer_bbox(frame)
    bbox = yolo.bbox_from_detect(results)
    if bbox is not None:
        if verbose:
            for idx, item in enumerate(bbox):
                print(f"Bounding box for detection #{idx} is {item}")
            print(f"Number of detected hands: {len(results)}")
    for idx, detection in enumerate(results):
        if verbose:
            print(f"Detection #{idx}: {detection}")
        idee, name, confidence, x, y, w, h = detection
        cx = x + (w / 2)
        cy = y + (h / 2)

        # draw a bounding box rectangle and label on the image
        if confidence > 0.8:
            color = (0, 255, 0)
        elif confidence > 0.6:
            color = (0, 255, 255)
        elif confidence > 0.5:            
            color = (0, 128, 255)            
        else:
            color = (0, 0, 255)
        cv2.rectangle(sframe, (x, y), (x + w, y + h), color, 2)
        #text = "%s (%s)" % (name, round(confidence, 2))
        #cv2.putText(sframe, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.5, color, 2)

    cv2.imshow("preview", sframe)

    rval, frame = vc.read()

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()
