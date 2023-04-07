import os, shutil, glob


def main():
    st_need = []
    for f_name in ['train', 'test', 'val']:
        with open(f_name + '.txt', 'r') as f:
            for i in f:
                for h in range(4):
                    st_need.append(i.strip()+'_'+str(h*90)+'.jpeg')

    st_all = os.listdir('./street_view_images')

    st_need = list(set(st_all).intersection(set(st_need)))
    st_rm = list(set(st_all).difference(set(st_need)))

    for s in st_rm:
        fn = os.path.join('./street_view_images', s)
        if os.path.exists(fn):
            os.remove(fn)


if __name__ == "__main__":
    main()
