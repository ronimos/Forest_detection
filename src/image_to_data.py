import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import cv2
import numpy as np
import pandas as pd
from config import DATA_PATH,IMAGE_PATH
from read_images import read_image_data

DATA_PATH = '../data'

ix, iy = -1, -1
xmin, ymin = -1, -1
drawing = False
zimg = np.zeros((100,100))
roi = ((-1,-1), (-1,-1))

def drwa_sample(event, x, y, flags, params):

    global ix, iy, zimg, drawing, marked_img, roi
    
    if event==cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y
        drawing = True
        
    if event == cv2.EVENT_MOUSEMOVE and drawing:
        marked_img = cv2.rectangle(zimg.copy(),
                                  (ix, iy),
                                  (x, y,),
                                  (0, 0, 255), 1)
    
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi = (min(xmin + ix, xmin + x), min(ymin + iy, ymin + y),
               max(xmin + ix, xmin + x), max(ymin + iy, ymin + y))
        
def gen_sample(img,
               window_name='Draw sample rect',
               mouse_callback_func=drwa_sample,
               pyr_levels=1):
    
    def zoom_in(zimg):
        return cv2.resize(zimg, None, fx=zoom, fy=zoom,
                          interpolation=cv2.INTER_AREA)

    global zimg, marked_img, xmin, ymin, roi
    zoom = 1
    #img = cv2.imread(path)
    for _ in range(pyr_levels):
        img = cv2.pyrDown(img)
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback_func)
    height, width, _ = img.shape
    xmin, ymin = 0, 0
    if width > 1500:
        xmin = np.random.choice(np.linspace(0, width, width//3).astype(int))
        ymin = np.random.choice(np.linspace(0, height, height//2).astype(int))
    if width > 8000:
        xmin = np.random.choice(np.linspace(0, width, width//6).astype(int))
        ymin = np.random.choice(np.linspace(0, height, height//6).astype(int))
    if width > 20000:
        xmin = np.random.choice(np.linspace(0, width, width//12).astype(int))
        ymin = np.random.choice(np.linspace(0, height, height//12).astype(int))
    
    zimg = img[ymin: min(height, ymin + 1000), 
               xmin: min(width, xmin+ 1500), 
              :].copy()
    marked_img = zimg.copy()
    while True:
        cv2.imshow(window_name, marked_img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("z"):
            zoom = zoom * 1.5
            zimg = zoom_in(zimg) 
            marked_img = zimg.copy()
        if k == ord("o"):
            zoom = zoom / 1.5
            zimg = zoom_in(zimg) 
            marked_img = zimg.copy()
        if k == ord('q'):
            cv2.destroyAllWindows()
            roi = (np.array(roi)/zoom).astype(int)
            break



def raster_dic_to_array(rasters,
                        pyr_levels=1):
    
    def pyr_down(img, pyr_levels):
        
        for _ in range(pyr_levels):
            img = cv2.pyrDown(img)
        return img

    ndvi = np.divide(rasters['NIR'] - rasters["Red"], rasters['NIR'] + rasters["Red"], 
                     out=np.zeros_like(rasters["NIR"]), 
                     where=rasters["NIR"] + rasters["Red"]!=0)
    
    imgs = [im for im in rasters.values()] + [ndvi]
    rasters_array = np.stack([pyr_down(im, pyr_levels) for im in imgs], axis=-1)
    
    return rasters_array

def gen_training_data(path,
                      imgs=None,
                      tr_type="Trees",
                      pyr_levels=1,
                      kernel=3):
        
    tr_dict = {"NoTrees": 0, "Trees": 1}
    if imgs is None:
        images, geo_transform = read_image_data()
        imgs = raster_dic_to_array(images)
    cordinates_lst = []
    with open(path) as f:
        for l in f.readlines():
            boundries = l.strip()
            try:
                xmin, ymin, xmax, ymax = map(int, boundries.split(","))
            except ValueError:
                continue
            vals = []
            roi = imgs[ymin: ymax, xmin: xmax]
            h, w, _ = roi.shape
            print(f"Type: {tr_type} ({w}, {h})")
            low_bound = kernel//2
            upper_bound = kernel - low_bound
            for x in range(low_bound, w-low_bound):
                for y in range(low_bound, h-low_bound):
                    vals.append(roi[y-low_bound:y+upper_bound, 
                                    x-low_bound:x+upper_bound, :].flatten())
            cordinates_lst = cordinates_lst + vals
    df = pd.DataFrame(cordinates_lst)
    df["target"] = tr_dict[tr_type]
    
    return df

def raster_to_table(path):
    
    rasters_dic, geo_transform = read_image_data(path)
    rastes_array = raster_dic_to_array(rasters_dic)
    n_rows, n_cols, _ = rastes_array.shape
    data_points = []
    for x in range(1, n_cols-1):
        for y in range(1, n_rows-1):
            point_data = rastes_array[y-1:y+2, x-1:x+2, :].flatten()
            data_points.append(point_data)
    data = pd.DataFrame(data_points)
    
    return data, geo_transform     

def result_to_raster(results, shape, geo_transform):
    
    new_shape = shape if len(results) == shape[0] * shape[1] else (shape[1] - 2, shape[0] - 2)
    results_img = results.reshape(new_shape).T    
    
    return results_img
    


if __name__ == "__main__":

    res = 5
    path = f'../data/images/Telluride/Telluride_{res}m.png'
    tr_type = "NoTrees"
    pyr_levels=0
    rois_path = f"../data/{tr_type}_{res}_m_res.txt"
    src_img = cv2.imread(path)
#    for i in range(7):
#        gen_sample(src_img, window_name=f"Draw No Trees Areas {i+1}/7",pyr_levels=pyr_levels)
#        with open(rois_path, "a") as f:
#            f.write(f"{','.join(roi.astype(str))}{os.linesep}")
            
    nt_df = gen_training_data(path=rois_path, 
                              imgs=src_img,
                              tr_type=tr_type,
                              pyr_levels=pyr_levels,
                              kernel=3)
    tr_type = "Trees"
    rois_path = f"../data/{tr_type}_{res}_m_res.txt"
    
    for i in range(7):
        gen_sample(src_img, window_name=f"Draw Trees Areas {i+1}/7",pyr_levels=pyr_levels)
        with open(rois_path, "a") as f:
            f.write(f"{','.join(roi.astype(str))}{os.linesep}")
    
    t_df = gen_training_data(path=rois_path,
                             imgs=src_img,
                             tr_type=tr_type,
                             pyr_levels=pyr_levels,
                             kernel=3)
    

    df = pd.concat([t_df, nt_df])
    df.to_csv(os.path.join(DATA_PATH, f"Test_train_data{res}_m_res.csv"), index=False)
