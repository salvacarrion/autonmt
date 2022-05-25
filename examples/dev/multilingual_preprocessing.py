import shutil
import random
random.seed(123)

from autonmt.bundle.utils import read_file_lines, write_file_lines


def add_tags(reverse=False):
    for lang1 in ["es", "fr", "de"]:
        for lang2 in ["en"]:
            for split in ["train"]:
                src = f"/home/scarrion/datasets/nn/translation/europarl_cf/{lang1}-en/original/data/raw"

                # Read files
                f_src = read_file_lines(f"{src}/{split}.{lang1}", autoclean=False)
                f_tgt = read_file_lines(f"{src}/{split}.{lang2}", autoclean=False)

                # Add tag
                if reverse:
                    f_src = [line for line in f_src]  # tgt
                    f_tgt = [f"<{lang2}-{lang1}> " + line for line in f_tgt]  # src
                    f_src, f_tgt = f_tgt, f_src
                else:
                    f_src = [f"<{lang1}-{lang2}> " + line for line in f_src]
                    f_tgt = [line for line in f_tgt]

                # # Save files
                write_file_lines(f_src, f"{src}/{split}.xx")
                write_file_lines(f_tgt, f"{src}/{split}.yy")
                print(f"Language files written: {lang1}-{lang2}-{split}")


def concat_tagged_files(shuffle=True):
    for ds_size in reversed(["original"]):
        datasets = {"train": {"src": [], "tgt": []}, "val": {"src": [], "tgt": []}, "test": {"src": [], "tgt": []}}

        split = "train"
        for lang1 in ["es", "fr", "de"]:
            for lang2 in ["en"]:
                src = f"/home/scarrion/datasets/nn/translation/europarl_cf/{lang1}-{lang2}/{ds_size}/data/raw"

                # Read files
                print(f"Reading files: {ds_size}-{split}")
                f_src = read_file_lines(f"{src}/{split}.xx", autoclean=False)
                f_tgt = read_file_lines(f"{src}/{split}.yy", autoclean=False)

                # Concat langs
                datasets[split]["src"] += f_src
                datasets[split]["tgt"] += f_tgt

        # Write multilingual files
        src, tgt = datasets[split]["src"], datasets[split]["tgt"]
        if shuffle:
            tmp = list(zip(src, tgt))
            random.shuffle(tmp)
            src, tgt = zip(*tmp)

        dst = f"/home/scarrion/datasets/nn/translation/europarl_cf/en-xx/{ds_size}/data/raw"
        write_file_lines(src, f"{dst}/{split}.en")
        write_file_lines(tgt, f"{dst}/{split}.xx")
        print(f"Multilingual files written: {ds_size}-{split}")
        ewr = 33


if __name__ == "__main__":
    # add_tags()
    concat_tagged_files()
    print("Done!")