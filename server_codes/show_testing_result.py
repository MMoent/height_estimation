import os


def main():
    exp_path = './Experiments/old'
    exp_dirs = os.listdir(exp_path)
    for d in exp_dirs:
        res_path = os.path.join(exp_path, d, 'test_result.txt')
        if not os.path.exists(res_path):
            continue
        res = ""
        with open(res_path, 'r') as f:
            res = f.read()

        print(d, res)


if __name__ == "__main__":
    main()
