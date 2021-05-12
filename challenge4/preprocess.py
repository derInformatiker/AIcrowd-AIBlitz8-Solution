import dataset, os

os.mkdir('data/preprocessed/train')
os.mkdir('data/preprocessed/val')
os.mkdir('data/preprocessed/test')
dataset.preprocess()