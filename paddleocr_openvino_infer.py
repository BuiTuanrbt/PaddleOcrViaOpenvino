import cv2
import numpy as np
import paddle
import math
import os
import json
import time
import random
import openvino as ov
import copy
import pre_post_processing as processing
import glob

from PIL import Image
from cus_logging import get_logger
from PIL import Image, ImageDraw, ImageFont


class PredictSystem(object):
    def __init__(self,det_model_path=None,det_weights_path= None,
                 rec_model_path=None,rec_weights_path=None, 
                 device = "CPU", drop_score=0.5,font_path = '/fonts/simfang.ttf', logger=None):
        
        
        if logger is None:
            self.logger = get_logger()
        self.drop_score = drop_score 
        self.device = device
        self.font_path = font_path
        core = ov.Core()
        self.det_model_path = det_model_path
        if self.det_model_path:
            self.det_model = core.read_model(det_model_path, det_weights_path)
            self.det_compiled_model = core.compile_model(model=self.det_model, device_name=self.device)
            self.det_input_layer = self.det_compiled_model.input(0)
            self.det_output_layer = self.det_compiled_model.output(0)
        self.rec_model_path = rec_model_path  
         
        if self.rec_model_path:
            rec_model = core.read_model(rec_model_path, rec_weights_path)

            # Assign dynamic shapes to every input layer on the last dimension.
            for input_layer in rec_model.inputs:
                input_shape = input_layer.partial_shape
                input_shape[3] = -1
                rec_model.reshape({input_layer: input_shape})

            self.rec_compiled_model = core.compile_model(model=rec_model, device_name="AUTO")
            # Get input and output nodes.
            self.rec_input_layer = self.rec_compiled_model.input(0)
            self.rec_output_layer = self.rec_compiled_model.output(0)

    def image_preprocess(self,input_image, size):
        """
        Preprocess input image for text detection

        Parameters:
            input_image: input image
            size: value for the image to be resized for text detection model
        """
        img = cv2.resize(input_image, (size, size))
        img = np.transpose(img, [2, 0, 1]) / 255
        img = np.expand_dims(img, 0)
        # NormalizeImage: {mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], is_scale: True}
        img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std
        return img.astype(np.float32)
    
    # Preprocess for text recognition.
    def resize_norm_img(self,img, max_wh_ratio):
        """
        Resize input image for text recognition

        Parameters:
            img: bounding box image from text detection
            max_wh_ratio: value for the resizing for text recognition model
        """
        rec_image_shape = [3, 48, 320]
        imgC, imgH, imgW = rec_image_shape
        assert imgC == img.shape[2]
        character_type = "ch"
        if character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im


    def prep_for_rec(self,dt_boxes, frame):
        """
        Preprocessing of the detected bounding boxes for text recognition

        Parameters:
            dt_boxes: detected bounding boxes from text detection
            frame: original input frame
        """
        ori_im = frame.copy()
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = processing.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        img_num = len(img_crop_list)
        # Calculate the aspect ratio of all text bars.
        width_list = []
        for img in img_crop_list:
            width_list.append(img.shape[1] / float(img.shape[0]))

        # Sorting can speed up the recognition process.
        indices = np.argsort(np.array(width_list))
        return img_crop_list, img_num, indices


    def batch_text_box(self,img_crop_list, img_num, indices, beg_img_no, batch_num):
        """
        Batch for text recognition

        Parameters:
            img_crop_list: processed detected bounding box images
            img_num: number of bounding boxes from text detection
            indices: sorting for bounding boxes to speed up text recognition
            beg_img_no: the beginning number of bounding boxes for each batch of text recognition inference
            batch_num: number of images for each batch
        """
        norm_img_batch = []
        max_wh_ratio = 0
        end_img_no = min(img_num, beg_img_no + batch_num)
        for ino in range(beg_img_no, end_img_no):
            h, w = img_crop_list[indices[ino]].shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
        for ino in range(beg_img_no, end_img_no):
            norm_img = self.resize_norm_img(img_crop_list[indices[ino]], max_wh_ratio)
            norm_img = norm_img[np.newaxis, :]
            norm_img_batch.append(norm_img)

        norm_img_batch = np.concatenate(norm_img_batch)
        norm_img_batch = norm_img_batch.copy()
        return norm_img_batch
    def post_processing_detection(self,frame, det_results):
        """
        Postprocess the results from text detection into bounding boxes

        Parameters:
            frame: input image
            det_results: inference results from text detection model
        """
        ori_im = frame.copy()
        data = {"image": frame}
        data_resize = processing.DetResizeForTest(data)
        data_list = []
        keep_keys = ["image", "shape"]
        for key in keep_keys:
            data_list.append(data_resize[key])
        img, shape_list = data_list

        shape_list = np.expand_dims(shape_list, axis=0)
        pred = det_results[0]
        if isinstance(pred, paddle.Tensor):
            pred = pred.numpy()
        segmentation = pred > 0.3

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            mask = segmentation[batch_index]
            boxes, scores = processing.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
            boxes_batch.append({"points": boxes})
        post_result = boxes_batch
        dt_boxes = post_result[0]["points"]
        dt_boxes = processing.filter_tag_det_res(dt_boxes, ori_im.shape)
        return dt_boxes
    def draw_box_txt_fine(self,img_size, box, txt, font_path="./fonts/simfang.ttf"):
        box_height = int(
            math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
        )
        box_width = int(
            math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        )

        if box_height > 2 * box_width and box_height > 30:
            img_text = Image.new("RGB", (box_height, box_width), (255, 255, 255))
            draw_text = ImageDraw.Draw(img_text)
            self.logger.info(f'Height& width img :{box_height},{box_width}')
            if txt:
                font = self.create_font(txt, (box_height, box_width), font_path)
                draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)
            img_text = img_text.transpose(Image.ROTATE_270)
        else:
            img_text = Image.new("RGB", (box_width, box_height), (255, 255, 255))
            draw_text = ImageDraw.Draw(img_text)
            if txt:
                font = self.create_font(txt, (box_width, box_height), font_path)
                draw_text.text([0, 0], txt, fill=(0, 0, 0), font=font)

        pts1 = np.float32(
            [[0, 0], [box_width, 0], [box_width, box_height], [0, box_height]]
        )
        pts2 = np.array(box, dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)

        img_text = np.array(img_text, dtype=np.uint8)
        img_right_text = cv2.warpPerspective(
            img_text,
            M,
            img_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return img_right_text
    def create_font(self,txt, sz, font_path="./fonts/simfang.ttf"):
        font_size = int(sz[1] * 0.99)
        # font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        font = ImageFont.load_default()
        
        # if int(PIL.__version__.split(".")[0]) < 10:
        #     length = font.getsize(txt)[0]
        # else:
        #     length = font.getlength(txt)

        # if length > sz[0]:
        #     font_size = int(font_size * sz[0] / length)
        #     font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
        return font

    def get_image_paths(self,input_path):
        """If input is an image file, return its path.
        If input is a directory, return all image file paths in the directory."""
        
        if os.path.isfile(input_path):
            return [input_path]
        
        elif os.path.isdir(input_path):
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp', '*.tiff']
            image_paths = []
            
            for ext in image_extensions:
                image_paths.extend(glob.glob(os.path.join(input_path, ext)))
            
            return image_paths
        
        else:
            print("The input path is neither a file nor a directory.")
            return []

    def draw_ocr_box_txt(self,
        image,
        boxes,
        txts=None,
        scores=None,
        drop_score=0.5,
        font_path=".fonts/simfang.ttf",
    ):
        h, w = image.height, image.width
        img_left = image.copy()
        img_right = np.ones((h, w, 3), dtype=np.uint8) * 255
        random.seed(0)

        draw_left = ImageDraw.Draw(img_left)
        if txts is None or len(txts) != len(boxes):
            txts = [None] * len(boxes)
        for idx, (box, txt) in enumerate(zip(boxes, txts)):
            if scores is not None and scores[idx] < drop_score:
                continue
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            draw_left.polygon(box, fill=color)
            img_right_text = self.draw_box_txt_fine((w, h), box, txt, font_path)
            pts = np.array(box, np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_right_text, [pts], True, color, 1)
            img_right = cv2.bitwise_and(img_right, img_right_text)
        img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(Image.fromarray(img_right), (w, 0, w * 2, h))
        return np.array(img_show)


    def predict_det(self,img_path):
        frame = cv2.imread(img_path)
        scale = 1280 / max(frame.shape)
        if scale < 1:
            frame = cv2.resize(
                src=frame,
                dsize=None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
        # Preprocess the image for text detection.
        test_image = self.image_preprocess(frame, 640)

        # Measure processing time for text detection.
        start_time = time.time()
        # Perform the inference step.
        det_results = self.det_compiled_model([test_image])[self.det_output_layer]
        dt_boxes = self.post_processing_detection(frame, det_results)
        dt_boxes = processing.sorted_boxes(dt_boxes)
        stop_time = time.time()
        elapse = stop_time-start_time
        if dt_boxes is None:
            self.logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
        else:
            self.logger.debug(
                "img path : {}, dt_boxes num : {}, elapsed : {}".format(img_path,len(dt_boxes), elapse)
            )
        return dt_boxes, frame, elapse
    def predict_rec(self, imgs_list,img_num, batch_num,indices):
        rec_res = [["", 0.0]] * img_num
        time_process_list ={
            "img_num": img_num,
            "batch_num": batch_num,
            "time_avg":0,
            "all_time": [],
            "i":0
        }
        all_time = []
        total_runtime_model = 0
        total_i = 0
        for beg_img_no in range(0, img_num, batch_num):
            total_i += 1
            norm_img_batch = self.batch_text_box(imgs_list, img_num, indices, beg_img_no, batch_num)

            st_batch = time.time()
            rec_results = self.rec_compiled_model([norm_img_batch])[self.rec_output_layer]
            _time =  time.time() - st_batch
            total_runtime_model += _time
            all_time.append(_time)
            # Postprocessing recognition results.
            postprocess_op = processing.build_post_process(processing.postprocess_params)
            rec_result = postprocess_op(rec_results)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
        time_process_list['time_avg'] = total_runtime_model/(total_i+0.00000000001)
        time_process_list['all_time'] = all_time
        time_process_list['i'] = total_i
        return rec_res, time_process_list
    
    def run(self,imgs,draw_img_save_dir,batch_num = 6,  is_visualize = True):
        batch_num = batch_num
        # save_file = img_path
        result = None
        total_det_res = []
        total_rec_res = []
        rec_img_list = []
        file_names = []
        imgs_path = self.get_image_paths(imgs)
        if self.det_model_path and self.rec_model_path:
            for img_path in imgs_path:
                # dt_boxes,frame, det_time = self.predict_det(img_path)
                det_res = self.predict_det(img_path)
                # dt_boxes = processing.sorted_boxes(dt_boxes)
                img_crop_list, img_num, indices = self.prep_for_rec(det_res[0], det_res[1])
                rec_res, detail_time = self.predict_rec(img_crop_list, img_num, batch_num, indices)
                self.logger.debug("imgs path: {}, rec_res num  : {}, time prcess detail : {}".format(img_path, len(rec_res), detail_time))
                total_det_res.append(det_res)
                total_rec_res.append(rec_res)
            result = total_det_res, total_rec_res
        else:
            if self.det_model_path:
                for img_path in imgs_path:
                    det_res = self.predict_det(img_path)
                    total_det_res.append(det_res)
                result = total_det_res,[]
            
            if self.rec_model_path:
                for rec_img_path in imgs_path:
                    rec_img_list.append(cv2.imread(rec_img_path))
                    file_name = os.path.basename(rec_img_path)
                    file_names.append(file_name.split('.')[0])
                rec_img_list = [cv2.imread(rec_img_path) for rec_img_path in imgs_path]
                img_num = len(rec_img_list)
                width_list = []
                for img in rec_img_list:
                    width_list.append(img.shape[1] / float(img.shape[0]))
                indices = np.argsort(np.array(width_list))
                rec_res, detail_time = self.predict_rec(rec_img_list, img_num, batch_num, indices)
                self.logger.debug("rec_res num  : {}, time prcess detail : {}".format(len(rec_res), detail_time))
                
                result = [], rec_res
            
        if is_visualize:
                if self.det_model_path and self.rec_model_path:
                    for det_res, rec_res,img_path in zip(total_det_res, total_rec_res, imgs_path):
                        image = Image.fromarray(cv2.cvtColor(det_res[1], cv2.COLOR_BGR2RGB))
                        save_file = img_path
                        boxes = det_res[0]
                        txts = [rec_res[i][0] for i in range(len(rec_res))]
                        scores = [rec_res[i][1] for i in range(len(rec_res))]
                        draw_img = self.draw_ocr_box_txt(
                            image,
                            boxes,
                            txts,
                            scores,
                            drop_score=self.drop_score,
                            font_path=self.font_path,
                        )
                        
                        cv2.imwrite(
                            os.path.join(draw_img_save_dir, os.path.basename(save_file)),
                            draw_img[:, :, ::-1],
                        )
                        self.logger.debug(
                            "The visualized image saved in {}".format(
                                os.path.join(draw_img_save_dir, os.path.basename(save_file))
                            )
                        )
                else:
                    if self.det_model_path:
                        
                        for det_res,img_path in zip(total_det_res, imgs_path):
                            image = Image.fromarray(cv2.cvtColor(det_res[1], cv2.COLOR_BGR2RGB))
                            save_file = img_path
                            img_det = image.copy()
                            file_name = os.path.basename(save_file)
                            file_name = file_name.split('.')[0]
                            draw_det = ImageDraw.Draw(img_det)
                            for idx, box in enumerate(det_res[0]):
                                # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                                draw_det.polygon(box, outline ="blue")
                            array_list_serializable = [arr.tolist() for arr in det_res[0]]
                            data = {"boxes":array_list_serializable}
                            with open(os.path.join(draw_img_save_dir, f"{file_name}_det_result.json"), 'w') as json_file:
                                json.dump(data, json_file, indent=4) 
                            img_det.save(os.path.join(draw_img_save_dir, f"{file_name}_det_result.png"))
                    if self.rec_model_path:
                        _, rec_res = result
                        txts = [rec_res[i][0] for i in range(len(rec_res))] 
                        for txt, rec_img,fn in zip(txts, rec_img_list, file_names):
                            cv2.imwrite(
                            os.path.join(draw_img_save_dir, f"{fn}  {txt}.png"),
                            rec_img,
                        )
                            self.logger.debug(
                            "The visualized image saved in {}".format(
                                os.path.join(draw_img_save_dir, f"{fn}  {txt}.png")
                            )
                                )
        return result
if __name__ == "__main__":
    base_url = "/Users/buituan/Work/MBBank/paddleocrViaOpenvino"     
    warmup = False  
    input_data = "/Users/buituan/Work/MBBank/PaddleSlim/my_test/data/img4.png"
    rec_data = "/Users/buituan/Work/MBBank/PaddleOCR/openvino_model/data_test/test_rec"
    
    predic_system = PredictSystem(
                                det_model_path=f"./model/det/inference.xml",
                                det_weights_path=f"./model/det/inference.bin",
                                rec_model_path=f"./model/rec/inference.xml",
                                rec_weights_path=f"./model/rec/inference.bin"
                                )
    if warmup:
        for i in range(10):
                result = predic_system.run(imgs=input_data, draw_img_save_dir="./result/ocr", is_visualize=False)
                 
    result = predic_system.run(imgs=input_data, draw_img_save_dir="./result/ocr", is_visualize=True)

