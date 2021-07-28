# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Xinzhi MU (draconids@sjtu.edu.cn)
# -----------------------------------------------------

import argparse
from time import sleep
from itertools import count
from tqdm import tqdm

import numpy as np
import torch
from reid import REID
import operator

from visualizer import AVAVisualizer
from action_predictor import AVAPredictorWorker

#pytorch issuse #973
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (rlimit[1], rlimit[1]))

def main():
    parser = argparse.ArgumentParser(description='Action Detection Demo')
    parser.add_argument(
        "--webcam",
        dest="webcam",
        help="Use webcam as input",
        action="store_true",
    )
    parser.add_argument(
        "--video-path",
        default="input.mp4",
        help="The path to the input video",
        type=str,
    )
    parser.add_argument(
        "--output-path",
        default="output.mp4",
        help="The path to the video output",
        type=str,
    )
    parser.add_argument(
        "--cpu",
        dest="cpu",
        help="Use cpu",
        action="store_true",
    )
    parser.add_argument(
        "--cfg-path",
        default="../config_files/resnet101_8x8f_denseserial.yaml",
        help="The path to the cfg file",
        type=str,
    )
    parser.add_argument(
        "--weight-path",
        default="../data/models/aia_models/resnet101_8x8f_denseserial.pth",
        help="The path to the model weights",
        type=str,
    )
    parser.add_argument(
        "--visual-threshold",
        default=0.5,
        help="The threshold of visualizer",
        type=float,
    )
    parser.add_argument(
        "--start",
        default=0,
        help="Start reading video at which millisecond",
        type=int,
    )
    parser.add_argument(
        "--duration",
        default=-1,
        help="The duration of detection",
        type=int,
    )
    parser.add_argument(
        "--detect-rate",
        default=4,
        help="Rate(fps) to update action labels",
        type=int
    )
    parser.add_argument(
        "--common-cate",
        default=False,
        help="Using common category model",
        action="store_true"
    )
    parser.add_argument(
        "--hide-time",
        default=False,
        help="Not show the timestamp at the corner",
        action="store_true"
    )
    parser.add_argument(
        "--tracker-box-thres",
        default=0.3,
        help="The box threshold for tracker",
        type=float,
    )
    parser.add_argument(
        "--tracker-nms-thres",
        default=0.6,
        help="The nms threshold for tracker",
        type=float,
    )

    args = parser.parse_args()

    args.input_path = 0 if args.webcam else args.video_path
    args.device = torch.device("cpu" if args.cpu else "cuda")
    args.realtime = True if args.webcam else False

    # Configuration for Tracker. Currently Multi-gpu is not supported
    args.gpus = "0"
    args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
    args.min_box_area = 0
    args.tracking = True
    args.detector = "tracker"
    args.debug = False

    reid = REID()
    print("ReID model loaded")

    if args.webcam:
        print('Starting webcam demo, press Ctrl + C to terminate...')
    else:
        print('Starting video demo, video path: {}'.format(args.video_path))

    # Initialise Visualizer
    video_writer = AVAVisualizer(
        args.input_path,
        args.output_path,
        args.realtime,
        args.start,
        args.duration,
        (not args.hide_time),
        confidence_threshold = args.visual_threshold,
        common_cate = args.common_cate,
    )

    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    ava_predictor_worker = AVAPredictorWorker(args)
    pred_done_flag = False

    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []
    
    print("Showing tracking progress bar (in fps). Other processes are running in the background.")
    try:
        for i in tqdm(count(), desc="Tracker Progress", unit=" frame"):
            with torch.no_grad():
                (frame, orig_img, boxes, scores, ids) = ava_predictor_worker.read_track()
                # print("type(frame) = {}".format(type(frame))) # <class 'numpy.ndarray'>

                if orig_img is None:
                    if not args.realtime:
                        ava_predictor_worker.compute_prediction()
                    break

                if args.realtime:
                    result = ava_predictor_worker.read()
                    flag = video_writer.realtime_write_frame(result, orig_img, boxes, scores, ids)
                    if not flag:
                        break
                else:
                    # print("demo.py - get images_by_id")
                    try:
                        ids_per_frame.append(set(map(int, ids)))
                        for bbox, id in zip(boxes, map(int, ids)):
                            if id not in images_by_id:
                                # images_by_id[id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                                images_by_id[id] = [frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]]
                            else:
                                # images_by_id[id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                                images_by_id[id].append(frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                        # for id in images_by_id.keys():
                        #     print("id: {0}, type(images_by_id[{0}]): {1}".format(id, type(images_by_id[id])))
                        #     last = len(images_by_id[id])-1
                        #     print("len(images_by_id[{}]): {}".format(id, last+1))
                        #     print("images_by_id[{}][{}].shape = {}".format(id, last, images_by_id[id][last].shape))
                    except TypeError:
                        pass
                    video_writer.send_track((boxes, ids))
                    while not pred_done_flag:
                        result = ava_predictor_worker.read()
                        if result is None:
                            break
                        elif result == "done":
                            pred_done_flag = True
                        else:
                            video_writer.send(result)
    except KeyboardInterrupt:
        print("Keyboard Interrupted")

    if not args.realtime:
        threshold = 320
        exist_ids = set()
        final_fuse_id = dict()

        print('Total IDs = ',len(images_by_id))
        feats = dict()
        for i in images_by_id:
            # print('ID number {} -> Number of frames {}'.format(i, len(images_by_id[i])))
            # print("type(images_by_id[{}] = {}".format(i, type(images_by_id[i])))
            # print("type(images_by_id[{}][0] = {}".format(i, type(images_by_id[i][0])))
            # print("type(images_by_id[{}]) = {}".format(i, type(images_by_id[i])))
            # print("type(images_by_id[{}][0] = {}".format(i, type(images_by_id[i][0])))
            try:
                feats[i] = reid._features(images_by_id[i])
                # print("type(feats[i]) = {}".format(type(feats[i])))
                # print("feats[i] = {}".format(feats[i]))
            except ValueError:
                pass

        for f in ids_per_frame:
            if f:
                if len(exist_ids) == 0:
                    for i in f:
                        final_fuse_id[i] = [i]
                    # print("type(exist_ids) = {}".format(type(exist_ids))) # set
                    # print("exist_ids = {}".format(exist_ids)) # {1, 2, 3, 4}, ...?
                    # print("type(f) = {}".format(type(f))) # set
                    # print("f = {}".format(f)) # {1, 2} ...?
                    exist_ids = exist_ids or f
                else:
                    new_ids = f-exist_ids
                    for nid in new_ids:
                        dis = []
                        if len(images_by_id[nid])<10:
                            exist_ids.add(nid)
                            continue
                        unpickable = []
                        for i in f:
                            for key,item in final_fuse_id.items():
                                if i in item:
                                    unpickable += final_fuse_id[key]
                        print('exist_ids {} unpickable {}'.format(exist_ids,unpickable))
                        for oid in (exist_ids-set(unpickable))&set(final_fuse_id.keys()):
                            try:
                                tmp = np.mean(reid.compute_distance(feats[nid],feats[oid]))
                                print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                                dis.append([oid, tmp])
                            except KeyError:
                                pass
                        exist_ids.add(nid)
                        print("type(nid) = {}".format(type(nid)))
                        print("nid = {}".format(nid))
                        if not dis:
                            final_fuse_id[nid] = [nid]
                            continue
                        dis.sort(key=operator.itemgetter(1))
                        if dis[0][1] < threshold:
                            combined_id = dis[0][0]
                            images_by_id[combined_id] += images_by_id[nid]
                            final_fuse_id[combined_id].append(nid)
                        else:
                            final_fuse_id[nid] = [nid]
        # print("demo.py - main()")
        # print("type(final_fuse_id) = {}".format(type(final_fuse_id)))
        print('Final ids and their sub-ids:', final_fuse_id)
        final_fuse_id_reverse = dict()
        for final_id, sub_ids in final_fuse_id.items():
            for sub_id in sub_ids:
                final_fuse_id_reverse[sub_id] = final_id
        # print("final_fuse_id_reverse = {}".format(final_fuse_id_reverse))

        # print('MOT took {} seconds'.format(int(time.time() - t1)))
        # t2 = time.time()

        video_writer.send_track("DONE")
        print("demo.py - main()")
        while not pred_done_flag:
            result = ava_predictor_worker.read()
            # print("type(result = ava_predictor_worker.read()) = {}".format(type(result)))
            # print("ava_predictor_worker.output_queue.get_nowait(): (predictions, center_timestamp, ids): result = {}".format(result))
            if result is None:
                sleep(0.1)
            elif result == "done":
                pred_done_flag = True
            else:
                for idx, id in enumerate(result[2]): # ids
                    id = int(id)
                    # print("BEFORE: result[2][{}] = {}".format(idx, result[2][idx]))
                    if id in final_fuse_id_reverse:
                        result[2][idx] = final_fuse_id_reverse[id]
                    # print("AFTER: result[2][{}] = {}".format(idx, result[2][idx]))
                video_writer.send(result)

        video_writer.send("DONE")
        tqdm.write("Showing video writer progress (in fps).")
        video_writer.progress_bar(i)

    video_writer.close()
    ava_predictor_worker.terminate()

if __name__ == "__main__":
    main()
