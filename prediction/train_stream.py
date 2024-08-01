import os
import random
import shutil

import config

from database.models import get_feedback_by_batch_id, get_unique_labels_from_table


def generate_dataset_dir(batch_id=0, data_split_ratio={"train": 0.8,"validation": 0.1,"test": 0.1,}):

    all_feedback = get_feedback_by_batch_id(batch_id)
    all_unique_labels = get_unique_labels_from_table("feedback")
    random.shuffle(all_feedback)
    splitted_feedback = {
        "info": {
            "count": {key: 0 for key in data_split_ratio.keys()},
            "error_file": [],
        }
    }
    total_count = len(all_feedback)
    for key, ratio in data_split_ratio.items():
        count = int(total_count * ratio)
        splitted_feedback["info"]["count"][key] = count
        splitted_feedback[key] = all_feedback[:count]
        all_feedback = all_feedback[:count]

    if not os.path.isdir(config.DATASET_BATCH_GENERATED):
        os.makedirs(config.DATASET_BATCH_GENERATED)

    batch_id_str = str(batch_id)
    dataset_batch_dir = os.path.join(config.DATASET_BATCH_GENERATED, batch_id_str)
    for dataset_type in data_split_ratio.keys():
        for label in all_unique_labels:
            if not os.path.isdir(
                os.path.join(dataset_batch_dir, dataset_type, label["label"])
            ):
                os.makedirs(
                    os.path.join(dataset_batch_dir, dataset_type, label["label"])
                )

    for dataset_type, feed_split in splitted_feedback.items():
        if dataset_type == "info":
            continue
        for feedback in feed_split:
            src = feedback["path"]
            dst = os.path.join(
                dataset_batch_dir,
                dataset_type,
                feedback["label"],
                os.path.basename(feedback["path"]),
            )
            if not os.path.isfile(src):
                splitted_feedback["info"]["error_file"].append(src)
                continue
            # os.symlink(src, dst)
            shutil.copyfile(src, dst)
    return {"dataset_batch_dir": dataset_batch_dir, "split": splitted_feedback}

def training():
    batch_id = 1
    dataset = generate_dataset_dir(batch_id)

    print(dataset)
    return dataset
# def g
