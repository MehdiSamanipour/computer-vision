import argparse
import os
import shutil

import kagglehub


def download_caltech256(target_dir: str = "") -> str:
    """
    Download Caltech256 using kagglehub and optionally copy to target_dir.
    Returns the usable dataset path.
    """
    source_path = kagglehub.dataset_download("jessicali9530/caltech256")
    print("KaggleHub cache path:", source_path)

    if not target_dir:
        return source_path

    target_dir = os.path.abspath(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    dataset_dir_name = os.path.basename(os.path.normpath(source_path))
    output_path = os.path.join(target_dir, dataset_dir_name)

    if os.path.exists(output_path):
        print("Target dataset path already exists:", output_path)
        return output_path

    shutil.copytree(source_path, output_path)
    print("Copied dataset to:", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download Caltech256 dataset via kagglehub.")
    parser.add_argument(
        "--target_dir",
        default="",
        help="Optional path where the dataset should be copied.",
    )
    args = parser.parse_args()
    final_path = download_caltech256(args.target_dir)
    print("Path to dataset files:", final_path)


if __name__ == "__main__":
    main()
