import argparse
from pathlib import Path
from typing import List, Union

import nibabel
import numpy
from PIL import Image
from PIL import ImageColor
from tqdm import tqdm


def id_to_color(class_id: float) -> numpy.ndarray:
    class_to_color_map = {
        "background": "#000000",
        "dimgray": "#696969",
        "forestgreen": "#228b22",
        "darkred": "#8b0000",
        "olive": "#808000",
        "lightseagreen": "#20b2aa",
        "darkblue": "#00008b",
        "red": "#ff0000",
        "darkorange": "#ff8c00",
        "yellow": "#ffff00",
        "lime": "#00ff00",
        "royalblue": "#4169e1",
        "deepskyblue": "#00bfff",
        "blue": "#0000ff",
        "fuchsia": "#ff00ff",
        "palevioletred": "#db7093",
        "khaki": "#f0e68c",
        "deeppink": "#ff1493",
        "lightsalmon": "#ffa07a",
        "violet": "#ee82ee",
    }
    assert class_id < len(class_to_color_map), "This script only support 20 different classes (including background)"
    return numpy.asarray(ImageColor.getrgb(list(class_to_color_map.values())[int(class_id)]))


def labels_to_color_image(label: numpy.ndarray, image_to_be_overlayed: Union[numpy.ndarray, None] = None) \
        -> numpy.ndarray:
    if image_to_be_overlayed is None:
        image = numpy.zeros((*label.shape, 3), dtype=numpy.uint8)
    else:
        image = (image_to_be_overlayed * 255).astype(numpy.uint8)
        if len(image.shape) == 2:
            image = numpy.repeat(numpy.expand_dims(image, axis=2), 3, axis=2)
    for class_id in numpy.unique(label):
        if class_id == 0.0:
            continue
        mask = numpy.where(label == class_id)
        image[mask] = id_to_color(class_id)
    return image


def get_case_ids_from_directory(directory: Path) -> List:
    return sorted(set([f.stem.split("_")[0][4:] for f in directory.iterdir()]))


def main(args: argparse.Namespace):
    assert args.result_dir.exists(), "Result directory doesn't exist."
    if len(args.case_ids) == 0:
        case_ids = get_case_ids_from_directory(args.result_dir)
    else:
        case_ids = args.case_ids
    for case_id in tqdm(case_ids, desc="Processing cases"):
        paths = [f"case{case_id}_img.nii.gz", f"case{case_id}_pred.nii.gz", f"case{case_id}_gt.nii.gz"]
        assert all([(args.result_dir / path).exists() for path in paths]), f"The required files for case id {case_id}" \
                                                                           f" do not exist. "
        data = [nibabel.load(args.result_dir / path).get_fdata() for path in paths]
        data = [numpy.transpose(arr, (2, 0, 1)) for arr in data]

        case_result_dir = args.result_dir / f"case{case_id}_resulting_images"
        case_result_dir.mkdir(parents=True, exist_ok=True)

        skipped_slices = []
        for slice_id, (original_image, prediction, gt) in enumerate(tqdm(zip(*data), desc="Processing slices", leave=False)):
            if args.skip_empty and not (gt.any() or prediction.any()):
                # there is nothing interesting on prediction and gt empty. Will be skipped
                skipped_slices.append(slice_id)
                continue
            if len(skipped_slices) > 1:
                print(f"Skipped slices {min(skipped_slices)}-{max(skipped_slices)}")
            if len(skipped_slices) == 1:
                print(f"Skipped slice {skipped_slices[0]}")
            skipped_slices = []

            prediction_image = labels_to_color_image(prediction, image_to_be_overlayed=original_image)
            gt_image = labels_to_color_image(gt, image_to_be_overlayed=original_image)
            combined_image = numpy.concatenate((gt_image, prediction_image), axis=1)

            out_img_filename = case_result_dir / f"case{case_id}_slice{slice_id:03d}.png"
            with open(out_img_filename, "wb") as out_img:
                Image.fromarray(combined_image).save(out_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path,
                        help="Path to the results directory that contains all the .nii.gz files")
    parser.add_argument("--skip-empty", action="store_true", default=False,
                        help="Only save those images where something was labelled in the groundtruth or in the "
                             "predicted image. (Avoids 'empty' images)")
    parser.add_argument("--case-ids", type=str, nargs="+", default=[],
                        help="List of case ids (with leading 0s) that should be processed")
    parsed_args = parser.parse_args()
    main(parsed_args)
