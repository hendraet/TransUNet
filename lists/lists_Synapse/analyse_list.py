if __name__ == '__main__':
    with open("train_original.txt", "r") as f:
        slices = f.readlines()
    case_ids = sorted(list(set([s.split("_")[0][4:] for s in slices])))
    print(case_ids)