import os
import cv2
import random


def main():
    data_dir = './Perspective'
    checked = []

    if os.path.exists('match.txt'):
        with open('match.txt', 'r') as f:
            for i in f:
                checked.append(i[:-4])

    with open('match.txt', 'a') as f:
        for i, d in enumerate(os.listdir(data_dir)):
            if d[:-4] in checked:
                continue
            path = os.path.join(data_dir, d)
            im = cv2.imread(path)
            cv2.imshow('1', im)
            c = cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(i+1, '/', len(os.listdir(data_dir)))
            if c == 13 or c == 32:     # accept
                f.write(d[:-4] + ' OK\n')
                print('accept')
            elif 48 <= c <= 57:
                f.write(d[:-4] + ' NO\n')
                print('reject')
            else:
                print('exit')
                break


if __name__ == "__main__":
    qualified_id = []
    with open('qualified.txt', 'r') as f:
        for i in f:
            qualified_id.append(i[:-1])

    # print(qualified_id)
    random.shuffle(qualified_id)
    train_idx, val_idx = int(len(qualified_id) * 0.7), int(len(qualified_id) * 0.85)
    train, val, test = qualified_id[:train_idx], qualified_id[train_idx:val_idx], qualified_id[val_idx:]
    with open('train.txt', 'w') as f:
        for i in train:
            f.write(i+'\n')
    with open('val.txt', 'w') as f:
        for i in val:
            f.write(i+'\n')
    with open('test.txt', 'w') as f:
        for i in test:
            f.write(i+'\n')
