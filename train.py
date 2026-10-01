"""Train the crop sorter from labeled image folders."""

import argparse

from src.crop_sorter.model import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train", help="Folder containing one subfolder per class")
    parser.add_argument("--model-out", default="artifacts/crop_sorter.joblib", help="Output path for the model")
    args = parser.parse_args()
    try:
        classes = train_model(args.data_dir, args.model_out)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved model to {args.model_out}")
    print("Classes: " + ", ".join(classes))


if __name__ == "__main__":
    main()
