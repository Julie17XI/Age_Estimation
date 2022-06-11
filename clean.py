import os
def clean():
    UTKFace_path = '/Users/xitang/Documents/CS6953/AIBias/data/UTKFace'
    original_path = '/Users/xitang/Documents/CS6953/AIBias/data/original'
    for filename in os.listdir(UTKFace_path):
        if filename.endswith(".jpg_flip.jpg"):
            file_path = os.path.join(UTKFace_path, filename)
            os.remove(file_path)
    for filename in os.listdir(original_path):
        if filename.endswith("jpg_train.tsv") or filename.endswith("jpg_test.tsv"):
            file_path = os.path.join(original_path, filename)
            os.remove(file_path)
        if filename == 'test_new.tsv' or filename == 'train_new.tsv':
            filename.truncate(0)

if __name__ ==  '__main__':
    clean()