import numpy as np
#
# 
# 
# import tensorflow as tf
import cv2
import time

from scipy import ndimage
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        #print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

def getFrame(sec):
    cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = cap.read()
    return hasFrames

if __name__ == "__main__":
    model_path = 'faster_rcnn/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture('out.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    (W, H) = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    result = cv2.VideoWriter('Results/resulttense4.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (W,H)) 
    sec = 0
    frameRate = 1.5
    success = getFrame(sec)
    #hasFrames,image = cap.read()
    #image = ndimage.rotate(image, 90)
    #image = cv2.resize(image, (848, 480))qq
    #cv2.imshow("G",image)
#rotation angle in degree
    framecount=0
    while success:
        r, img = cap.read()
        print("test")
        #img = cv2.resize(img, (424, 240))
        #img = cv2.resize(img, (1920, 1080))
        #img = ndimage.rotate(img, 90)
        if r:

            boxes, scores, classes, num = odapi.processFrame(img)
            
            final_score = np.squeeze(scores)    
            count = 0
                
            # Visualization of the results of a detection.
            for i in range(len(boxes)):
                # Class 1 represents human
                if scores is None or final_score[i] > threshold:
                    count = count + 1
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,"Person detected {}".format(str(count)),(10,50), font, 0.75,(255,0,0),1,cv2.LINE_AA)
            #cv2.imshow("preview", img)
            result.write(img)
            #sec = sec + frameRate
            #sec = round(sec, 2)
            #success = getFrame(sec)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        framecount+=1
        if framecount==total:
            break

cap.release()
result.release()
cv2.destroyAllWindows()