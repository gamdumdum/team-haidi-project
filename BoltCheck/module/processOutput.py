
class processOutput:
    @staticmethod
    def process_output(boxes_output, labels_output, frame_shape, W_model, H_model):
        boxes = []
        h, w = frame_shape[:2]

        for i in range(boxes_output.shape[1]):
            conf = boxes_output[0, i, 4]
            if conf > 0.3:
                # 좌표 클리핑 및 스케일링
                x1 = max(0, min(boxes_output[0, i, 0], W_model-1))
                y1 = max(0, min(boxes_output[0, i, 1], H_model-1))
                x2 = max(0, min(boxes_output[0, i, 2], W_model-1))
                y2 = max(0, min(boxes_output[0, i, 3], H_model-1))
                
                x1 = int(x1 * (w / W_model))
                x2 = int(x2 * (w / W_model))
                y1 = int(y1 * (h / H_model))
                y2 = int(y2 * (h / H_model))
                
                x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
                y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))
                
                label = int(labels_output[0, i])
                boxes.append((x1, y1, x2, y2, conf, label))
        
        return boxes