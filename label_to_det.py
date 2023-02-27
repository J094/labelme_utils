#!/usr/bin/env python3
# J094
# 2023.02.15

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import labelme


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input json directory")
    parser.add_argument("output_dir", help="output annotation directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()
    
    if osp.exists(args.output_dir):
        print("Output directory already exists: ", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "viz"))
    print("Creating annotations: ", args.output_dir)
    
    class_names = []
    for line in open(args.labels).readlines():
        class_name = line.strip()
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("Class names: ", class_names)
    
    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating annotations from: ", filename)

        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]

        out_txt_file = osp.join(args.output_dir, base + ".txt")
        if not args.noviz:
            out_viz_file = osp.join(
                args.output_dir, "viz", base + ".png"
            )

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        [h, w] = img.shape[:2]
        # print(h, w)

        bboxes = []
        labels = []
        for shape in label_file.shapes:
            if shape["shape_type"] != "rectangle":
                print(
                    "Skipping shape: label={label}, "
                    "shape_type={shape_type}".format(**shape)
                )
                continue

            class_name = shape["label"]
            class_id = class_names.index(class_name)

            (xmin, ymin), (xmax, ymax) = shape["points"]
            # swap if min is larger than max.
            xmin, xmax = sorted([xmin, xmax])
            ymin, ymax = sorted([ymin, ymax])

            bboxes.append([ymin, xmin, ymax, xmax])
            labels.append(class_id)
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            bbox_w = xmax - xmin + 1
            bbox_h = ymax - ymin + 1
            
            bbox_to_write = str(class_id) + " " + str(center_x / w) + " " + str(center_y / h) + " " + str(bbox_w / w) \
                            + " " + str(bbox_h / h) + "\n"
            
            with open(out_txt_file, "a") as f:
                f.writelines(bbox_to_write)
        with open(out_txt_file, "a") as f:
            f.writelines(" ")

        if not args.noviz:
            captions = [class_names[label] for label in labels]
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                bboxes=bboxes,
                captions=captions,
                font_size=15,
            )
            imgviz.io.imsave(out_viz_file, viz)
            

if __name__ == "__main__":
    main()