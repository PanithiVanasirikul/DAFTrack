import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
import json
from loguru import logger
import imageio.v2 as imageio  
import sys

from f_utils.visualize import plot_tracking,plot_r_tracking
from f_utils.timer import Timer

# from tracker.RF_tracker import RFTrack
from tracker.UnscentedGroundRF_tracker import UnscentedGroundRFTrack
from tracker.UnscentedGroundRF_tracker import f_world, h_world_to_pixel, pix_to_world, world_to_pix
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints

from f_utils.iou import AssociationFunction
import motmetrics as mm


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
# ROT_TRACKERS = ['RFTrackB','RFTrackK',RFTrack]

def make_parser():
    parser = argparse.ArgumentParser("RF-TRACKER Demo!")

    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
        
    parser.add_argument("--cut_off", type=int, default=0, help="early cut off for experimentation")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.35, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    
    parser.add_argument("--match", default="gwd", help="Association type, e.g gwd, bd, kld")

    #dataset arg
    parser.add_argument("--dataset", default="CEPDOF", help="CEPDOF scenes")
    # parser.add_argument("--dataset", default="WEPDTOF", help="WEPDTOF scenes")

    parser.add_argument("--match1", default=150, help="default Euclidean distance for CEPDOF, 150 cm")
    # parser.add_argument("--match1", default=75, help="default Euclidean distance for WEPDTOF, 75 cm")

    parser.add_argument("--use_camera_height", default=False, help="scale Euclidean distances with camera height or not")

    # parser.add_argument("--camera_height", default=1.7, help="camera height used. using camera_height = -1 means using camera height estimation")
    parser.add_argument("--camera_height", default=-1, help="camera height used. using camera_height = -1 means using camera height estimation")

    return parser

def vid_to_imglist(vid_path):
    output_folder = 'demo_frames'  
    os.makedirs(output_folder, exist_ok=True)  
    
    print("Transforming video to folder of frames")
    # Read the GIF  
    decoded_vid = imageio.get_reader(vid_path)  
    
    # Iterate through each frame and save it as an image  
    for i, frame in enumerate(decoded_vid):   
        imageio.imsave(os.path.join(output_folder, f"demo_{i+1:03d}.png"), frame)

def get_image_list(path):
    image_paths = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_paths.append(apath)
    return image_paths

def get_frame_info(image_path):
    frame = cv2.imread(image_path)
    height, width, channels = frame.shape
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 0.0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    # Compute minimum enclosing circle
    center = (0.0,0.0)
    radius = 0.0
    if largest_contour is not None:
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (x, y)
        radius = radius
    return frame, height, width, center, radius

def get_img_info(image_path,label_map, read_img=True):
    
    img_num = image_path.split('.')[0]
    img_num = img_num.split(os.path.sep)[-1]

    if img_num in label_map:
        labels = label_map[img_num]
        labels = np.array(labels).astype(np.float64)
    else:
        labels = None
    if read_img:
        frame, height, width, _, _ = get_frame_info(image_path)
        info = {
            'raw_img': frame,
            "height": height,
            "width" :width,
        }
    else:
        info = None

    return labels,info

def create_det_labels(path,with_rot=True):
    with open(path,"r") as f:
        labels = json.load(f)
    lbl_map = {}

    for lbl in labels:
        frame = lbl["img_name"]
        res = []
        
        for bbox,score in zip(lbl["bboxes"].copy(), lbl["scores"]):
            if with_rot:
                assert len(bbox) == 5 
            else:
                assert len(bbox) == 4

            res.append(bbox + [score]) 
        
        if res != []:
            if frame in lbl_map:
                lbl_map[frame ] += res
            else:
                lbl_map[frame ] = res
    return lbl_map 

def store_res(frame_id,track,center,radius,with_rot=True,min_box_area=10,):

    boxes,ids,scores = [],[],[]
    res = []


    box = track.bbox if with_rot else track.tlwh
    tid = track.track_id
    score = track.score
    mean = track.covMatrix[0]
    cov = track.covMatrix[1]

    points_world_to_pix = MerweScaledSigmaPoints(n=2, alpha=.1, beta=2., kappa=0.)
    sigma = points_world_to_pix.sigma_points(mean, cov)
    sigma_transformed = world_to_pix(sigma)
    mu_cov_pixel = unscented_transform(sigma_transformed, points_world_to_pix.Wm, points_world_to_pix.Wc, residual_fn=np.subtract)
    mu_pixel = (mu_cov_pixel[0]*radius) + center
    cov_pixel = (mu_cov_pixel[1]*radius**2)
    mu_world_future = f_world(track.x, 4)[[0, 2]]
    mu_pixel_future = (world_to_pix(mu_world_future)*radius) + center
    

    if box[2] * box[3] > min_box_area:
        boxes = [box]
        ids = [tid]
        scores = [score]
        mu_pixels = [mu_pixel]
        cov_pixels = [cov_pixel]
        mu_pixels_future = [mu_pixel_future]
        
        # save results in a txt format
        if with_rot:
            save_format = '{frame},{id},{x1},{y1},{w},{h},{r},{s},-1,-1,-1\n'
            x1,y1,w,h,r = box
            res.append(
                save_format.format(frame=frame_id, id=tid, 
                                x1=round(x1, 1), y1=round(y1, 1), 
                                w=round(w, 1), h=round(h, 1),
                                r=round(r, 1), s=round(score,4))
            )
        else:

            save_format = '{frame},{id},{x},{y},{w},{h},0,{s},-1,-1,-1\n'
            t,l,w,h = box

            x = t + w/2
            y = l + h/2

            res.append(
                save_format.format(frame=frame_id, id=tid, 
                                x=round(x, 1), y=round(y, 1), 
                                w=round(w, 1), h=round(h, 1), s=round(score,4))
            )

    return boxes,ids,scores,res,mu_pixels,cov_pixels,mu_pixels_future


def track(out_folder,
            img_path,
            det_path,
            gt_path,
            args,
            out_name = None,
            frame_start = 0,
            with_rot=True,
            out_video = False,
            out_images = False):
    output_json = list()
    print(f"Tracking on {os.path.basename(img_path)}")
    files = get_image_list(img_path)

    files.sort()

    with open(gt_path,"r") as f:
        gt_json = json.load(f)
    
    gt_json_dict = dict()
    for i in range(len(gt_json)):
        gt_json_dict[int(gt_json[i]['img_name'].split("_")[-1]) - int((files[0].split("_")[-1]).split(".jpg")[0])] = gt_json[i]

    if out_name is None:
        current_time = time.localtime()
        out_name = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)

    images_folder = osp.join(out_folder, 'images')
    os.makedirs(images_folder, exist_ok=True)
    video_folder = osp.join(out_folder, 'video')
    os.makedirs(video_folder, exist_ok=True)
    det_folder = osp.join(out_folder, 'dets')
    os.makedirs(det_folder, exist_ok=True)
    
    _, height, width, center, radius = get_frame_info(files[0])
    if out_video:
        save_path = osp.join(video_folder, f"{out_name}.mp4")
        
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (int(width), int(height))
        )

    args.frame_size = [int(height),int(width)]

    print(f"Frame size: width: {int(width)} , height:{int(height)}")
    print(f"Association method : {args.match}")
    
  
    tracker = UnscentedGroundRFTrack(args, center, radius)
    f_frame = osp.basename(files[0]).split('_')[-1].split('.')[0]
    f_frame = (int(f_frame))

    timer = Timer()
    results = []

    print("Making labels")  

    # Create a hashmap of all the detection in each frame
    lbl_map = create_det_labels(det_path,with_rot=with_rot)

    # Break early if there is a cut off argument
    cut_off = True if args.cut_off != 0 else False

    print("Starting...")

    for frame_id, img_path in enumerate(files, frame_start):
        timer.tic()
        outputs, img_info = get_img_info(img_path, lbl_map, read_img=(out_video or out_images))

        if cut_off:
            if frame_id == args.cut_off : break

        real_f_id = frame_id + f_frame
        if outputs is not None:
            # update the tracker with new bounding box frames
            online_targets = tracker.update(outputs)
            
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_mus = []
            online_covs = []
            online_mus_future = []

            #  store the new tracking results
            for t in online_targets:
                tboxes,tids,tscores,tres,tmu,tcov,tmu_future = store_res(real_f_id, t, center, radius, with_rot,args.min_box_area)
                online_tlwhs += tboxes
                online_mus += tmu
                online_covs += tcov
                online_mus_future += tmu_future
                online_ids += tids
                online_scores += tscores
                results += tres
            output_json.append({"img_name": os.path.splitext(os.path.basename(img_path))[0],
                                "bboxes": [e.tolist() for e in online_tlwhs],
                                "scores": online_scores,
                                "id": online_ids})   
        
        else:
            output_json.append({"img_name": os.path.splitext(os.path.basename(img_path))[0],
                                "bboxes": [],
                                "scores": [],
                                "id": []})

        timer.toc()
        if out_video or out_images:
            if outputs is not None:
                drawer = plot_r_tracking if with_rot else plot_tracking
                online_im = drawer(img_info['raw_img'], online_tlwhs, online_mus, online_covs, online_mus_future, online_ids, frame_id=real_f_id, fps= 1./timer.average_time)
            else:
                online_im = img_info['raw_img']
            if out_video:
                vid_writer.write(online_im)
            if out_images:
                cv2.imwrite(os.path.join(images_folder, f"{os.path.splitext(os.path.basename(img_path))[0]}_{tracker.frame_id}.jpg"), online_im)
                cv2.imwrite(os.path.join(images_folder, "latest.jpg"), online_im)
        # breakpoint()

        if frame_id % 150 == 0:
            logger.info('Processing frame {} ({:.5f} fps)'.format(real_f_id, 1/max(1e-5, timer.average_time)))

    if out_video:
        vid_writer.release()
        logger.info(f"Results saved at {osp.join(video_folder, out_name)}.mp4")

    res_file = os.path.join(det_folder, f"{out_name}.json")
    with open(res_file, "w") as f:
        json.dump(output_json, f)

    assoc = AssociationFunction(width, height, "iou_obb")
    acc = mm.MOTAccumulator()
    
    for frame_id, output_per_frame in enumerate(output_json):
        frame_id_scale = frame_id
        if frame_id_scale not in gt_json_dict:
            C = []
            acc.update([], output_per_frame['id'], C, frameid=frame_id)
        elif len(output_per_frame['id']) == 0:
            gt_per_frame = gt_json_dict[frame_id_scale]
            C = []
            acc.update(gt_per_frame['id'], output_per_frame['id'], C, frameid=frame_id)
        else:
            gt_per_frame = gt_json_dict[frame_id_scale]
            C = 1 - assoc.asso_func(gt_per_frame['bboxes'], output_per_frame['bboxes'])
            acc.update(gt_per_frame['id'], output_per_frame['id'], C, frameid=frame_id)

    return acc

def single_run(args, path, det_path, gt_path):
    vis_folder = osp.join(os.getcwd(), "track_vis")
    os.makedirs(vis_folder, exist_ok=True)
    acc = track(vis_folder, path, det_path, gt_path, args = args, with_rot = True)
    return acc

if __name__ == "__main__":
    args = make_parser().parse_args()

    if args.dataset == "CEPDOF":
        CEPDOF_images_dir = "/mnt/ssd1/datasets/fisheye_tracking_datasets/CEPDOF"
        CEPDOF_images_folders = [os.path.join(CEPDOF_images_dir, e) for e in os.listdir(CEPDOF_images_dir) if e!="annotations"]
        CEPDOF_annotations_dir = "./formatted_jsons/ground_truth/CEPDOF"
        CEPDOF_annotations_files = [os.path.join(CEPDOF_annotations_dir, e) for e in os.listdir(CEPDOF_annotations_dir)]
        # CEPDOF_prediction_dir = "/mnt/ssd1/datasets/fisheye_tracking/my_own_pipeline/formatted_jsons/predictions_608/CEPDOF"
        CEPDOF_prediction_dir = "./formatted_jsons/predictions_1024/CEPDOF"
        # CEPDOF_prediction_dir = "/mnt/ssd1/datasets/fisheye_tracking/my_own_pipeline/formatted_jsons/ground_truth/CEPDOF"
        CEPDOF_prediction_files = [os.path.join(CEPDOF_prediction_dir, e) for e in os.listdir(CEPDOF_prediction_dir)]

        paths = CEPDOF_images_folders
        gt_paths = CEPDOF_annotations_files
        det_paths = CEPDOF_prediction_files

    elif args.dataset == "WEPDTOF":
        WEPDTOF_images_dir = "/mnt/ssd1/datasets/fisheye_tracking_datasets/WEPDTOF/frames"
        WEPDTOF_images_folders = [os.path.join(WEPDTOF_images_dir, e) for e in os.listdir(WEPDTOF_images_dir)]
        WEPDTOF_annotations_dir = "./ground_truth/WEPDTOF"
        WEPDTOF_annotations_files = [os.path.join(WEPDTOF_annotations_dir, e) for e in os.listdir(WEPDTOF_annotations_dir)]
        # WEPDTOF_prediction_dir = "/mnt/ssd1/datasets/fisheye_tracking/my_own_pipeline/formatted_jsons/predictions_608/WEPDTOF"
        WEPDTOF_prediction_dir = "./predictions_1024/WEPDTOF"
        # WEPDTOF_prediction_dir = "/mnt/ssd1/datasets/fisheye_tracking/my_own_pipeline/formatted_jsons/ground_truth/WEPDTOF"
        WEPDTOF_prediction_files = [os.path.join(WEPDTOF_prediction_dir, e) for e in os.listdir(WEPDTOF_prediction_dir)]

        paths = WEPDTOF_images_folders
        gt_paths = WEPDTOF_annotations_files
        det_paths = WEPDTOF_prediction_files

    paths.sort()
    gt_paths.sort()
    det_paths.sort()

    all_accs = list()
    for path, det_path, gt_path in zip(paths, det_paths, gt_paths):
        acc = single_run(args, path, det_path, gt_path)
        all_accs.append(acc)
    
    mh = mm.metrics.create()
    summary = mh.compute_many(all_accs, metrics=["deta_alpha", "mota", "hota_alpha", "idf1", "assa_alpha", "num_switches"], names=[os.path.basename(e) for e in paths], generate_overall=True)
    print("\n")
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
    print(strsummary)
