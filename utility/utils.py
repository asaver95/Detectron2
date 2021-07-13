import  numpy as np
import  cv2
import matplotlib.pyplot as plt
import json

def getTile(im,cx,cy,w,h):
    tile = im[ cy-h//2:cy+h//2, cx-w//2:cx+w//2]
    return tile

def AppendBoxes(BBOX, boxes,CX,CY,w,h):
    for bbox in boxes: 
        bbox = np.array(bbox.to("cpu"))
        # Changing the reference frame:
        x1 = CX - w//2 +  float(bbox[0])
        y1 = CY - h//2 +  float(bbox[1])
        x2 = CX - w//2 +  float(bbox[2])
        y2 = CY - h//2 +  float(bbox[3])
        # Updating the coordinates of bbox:
        bbox_full_frame = [int(x1),int(y1),int(x2),int(y2)]     # In the reference of the full image 
        BBOX.append(bbox_full_frame)
    return BBOX

def show_tile():
    v = Visualizer(tile[:, :, ::-1],
                       scale=1, 
                       instance_mode=ColorMode.IMAGE_BW   
                       # remove the colors of unsegmented pixels. Only available for segmentation 
        )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(10,10))
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()



def draw_boxes(image, BBOX):
    for b in BBOX:
        # Top left corner of rectangle
        start_point = (b[0], b[1])
        # Bottom right corner of rectangle
        end_point = (b[2], b[3])
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 15
        # Draw a rectangle
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


def clahe_apply(tile):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
    equ = clahe.apply(tile[:,:,1])
    image = cv2.cvtColor(equ, cv2.COLOR_GRAY2RGB)
    return image

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def plt_model(model_dir):    
    root = 'output/'+ model_dir + '/metrics.json'
    experiment_metrics = load_json_arr(root)
 
    fig, ax1 = plt.subplots(dpi=300, figsize=(7,5))

    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')

    ax1.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x], color='black',ls='-', label="Total Loss")
    ax1.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x], color="dimgray", ls='--' ,label="Val Loss")
        
    ax1.tick_params(axis='y')
    plt.legend(loc='upper left')

    # fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    color = 'tab:orange'
    ax2.set_ylabel('AP')
    ax2.plot(
        [x['iteration'] for x in experiment_metrics if 'bbox/AP' in x], 
        [x['bbox/AP'] for x in experiment_metrics if 'bbox/AP' in x], color='black',ls='-.', label="AP")
    ax2.tick_params(axis='y')
    plt.legend(loc='upper right')

    AP = [x['bbox/AP'] for x in experiment_metrics if 'bbox/AP' in x]
    # plt.title(model_dir +'           ('+'$mAP_{max}$ =' + f' {np.nanmax(AP):.2f}%'+')', fontweight='bold')
    plt.savefig('output/'+ model_dir +'/fig.png')
    plt.show()





if __name__ == "__main__":
   pass