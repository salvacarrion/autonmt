import shutil
import random
random.seed(123)

from autonmt.bundle.utils import read_file_lines, write_file_lines


def add_tags():
    for lang1 in ["es", "fr", "de"]:
        for lang2 in ["en"]:
            for split in ["train"]:
                src = f"/home/scarrion/datasets/nn/translation/europarl_cf/{lang1}-en/original/data/raw"

                # Read files
                f_src = read_file_lines(f"{src}/{split}.{lang1}", autoclean=False)
                f_tgt = read_file_lines(f"{src}/{split}.{lang2}", autoclean=False)

                # Add tag
                f_src = [f"<{lang1}> " + line for line in f_src]
                f_tgt = [f"<{lang2}> " + line for line in f_tgt]

                # # Save files
                write_file_lines(f_src, f"{src}/{split}.tagged.{lang1}")
                write_file_lines(f_tgt, f"{src}/{split}.tagged.{lang2}")
                print(f"Language files written: {lang1}-{split}")


def concat_tagged_files(shuffle=True):
    for ds_size in reversed(["original", "500k", "100k", "10k"]):
        datasets = {"train": {"src": [], "tgt": []}, "val": {"src": [], "tgt": []}, "test": {"src": [], "tgt": []}}

        for lang1 in ["es", "fr", "de"]:
            for lang2 in ["en"]:
                for split in ["train", "val", "test"]:
                    src = f"/home/scarrion/datasets/nn/translation/europarl_cf/{lang1}-{lang2}/{ds_size}/data/splits"

                    # Read files
                    print(f"Reading files: {ds_size}-{split}")
                    f_src = read_file_lines(f"{src}/{split}.{lang1}", autoclean=False)
                    f_tgt = read_file_lines(f"{src}/{split}.{lang2}", autoclean=False)

                    # Concat langs
                    datasets[split]["src"] += f_src
                    datasets[split]["tgt"] += f_tgt

        # Write multilingual files
        for split in ["train", "val", "test"]:
            src, tgt = datasets[split]["src"], datasets[split]["tgt"]
            if shuffle:
                tmp = list(zip(src, tgt))
                random.shuffle(tmp)
                src, tgt = zip(*tmp)

            dst = f"/home/scarrion/datasets/nn/translation/europarl_cf/xx-yy/{ds_size}/data/splits"
            write_file_lines(src, f"{dst}/{split}.xx")
            write_file_lines(tgt, f"{dst}/{split}.yy")
            print(f"Multilingual files written: {ds_size}-{split}")
        ewr = 33


if __name__ == "__main__":
    # add_tags()
    # concat_tagged_files()
    print("Done!")