#!/usr/bin/env python3
# J094
# 2023.02.15

import argparse
import glob
import os
import os.path as osp
import sys
import uuid
import math

import imgviz
import numpy as np
import PIL.Image
import PIL.ImageDraw

import labelme


def shape_to_mask(
    img_shape, points, shape_type=None, line_width=10, point_size=5
):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if cls_name in label_name_to_value:
            if instance not in instances:
                instances.append(instance)
            ins_id = instances.index(instance) + 1
            cls_id = label_name_to_value[cls_name]

            mask = shape_to_mask(img_shape[:2], points, shape_type)
            cls[mask] = cls_id
            ins[mask] = ins_id

    return cls, ins

def lblsave(filename, lbl, color_map):
    if osp.splitext(filename)[1] != ".png":
        filename += ".png"
    # Assume label ranses [-1, 254] for int32,
    # and [0, 255] for uint8 as VOC.
    if lbl.min() >= -1 and lbl.max() < 255:
        lbl_pil = PIL.Image.fromarray(lbl.astype(np.uint8), mode="P")
        lbl_pil.putpalette(color_map.flatten())
        lbl_pil.save(filename)
    else:
        raise ValueError(
            "[%s] Cannot save the pixel-wise class label as PNG. "
            "Please consider using the .npy format." % filename
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="input json directory")
    parser.add_argument("output_dir", help="output annotation directory")
    parser.add_argument("--labels", help="labels file", required=True)
    parser.add_argument("--palette", help="palette file", required=True)
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
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i # starts with 0
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("Class names: ", class_names)
    
    color_map = np.empty(shape=(len(class_names), 3), dtype=np.uint8)
    for i, line in enumerate(open(args.palette).readlines()):
        class_id = i
        class_palette = eval(line.strip())
        color_map[i] = class_palette
    
    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating annotations from: ", filename)
        
        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]
        
        out_img_file = osp.join(args.output_dir, base + ".png")
        if not args.noviz:
            out_viz_file = osp.join(args.output_dir, "viz", base + ".png")
        
        img = labelme.utils.img_data_to_arr(label_file.imageData)
        lbl, _ = shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        lblsave(out_img_file, lbl, color_map)
        
        if not args.noviz:
            viz = imgviz.label2rgb(
                lbl,
                imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb",
                colormap=color_map,
            )
            imgviz.io.imsave(out_viz_file, viz)


if __name__ == "__main__":
    main()
