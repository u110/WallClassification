import os
import glob
import crop
import argparse as ap
import PIL
from functools import reduce
import tqdm


# root_dir下のファイルパスを全列挙
def walk(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            yield os.path.join(root, file)


def concat(lst, start):
    return reduce(lambda x, y: x+y, lst, start)


def isImg(path):
    ext = os.path.splitext(os.path.basename(path))[1].lower()
    # print("ext=",ext)
    return ext in [".jpg", ".jpeg", ".png"]


if __name__ == "__main__":
    parser = ap.ArgumentParser(description="resize")
    parser.add_argument('inputFile', help='Input file path', nargs="+")
    parser.add_argument("-o", '--outputPath', help='Output file path', default="./")
    parser.add_argument("-s", '--size', help='size', default="256", type=int)
    parser.add_argument("-r", "--recursive", help="inputFileディレクトリ以下を再帰的に走査する", action="store_true")
    parser.add_argument("-c", "--crop_max_square", help="長方形画像の場合に、中心最大正方形を切り取る処理を入れる", action="store_true")
    args = parser.parse_args()

    if args.recursive:
        imgPaths = args.inputFile
    else:
        imgPaths = list(filter(isImg, reduce(lambda x, y: x+y, [glob.glob(arg) for arg in args.inputFile])))

    print(imgPaths)

    if args.recursive:
        imgPaths = [(root, paths) for root, paths in zip(imgPaths, [list(walk(path)) for path in imgPaths])]
    else:
        imgPaths = [(os.path.dirname(path), [path]) for path in imgPaths]

    if not os.path.exists(args.outputPath):
        os.mkdir(args.outputPath)

    total = len(concat([paths for _, paths in imgPaths], []))
    print(total, len(imgPaths))
    bar = tqdm.tqdm(total=total)
    for root, paths in imgPaths:
        for imgPath in paths:
            try:
                bar.update()
                img = PIL.Image.open(imgPath, mode="r")
                if args.crop_max_square:
                    img = crop.crop_max_square(img)
                img = img.resize((args.size, args.size), resample=PIL.Image.BILINEAR)
                filepath, _ = os.path.splitext(os.path.relpath(imgPath, root))
                # filename = os.path.dirname(imgPath) + "_" + filename
                if not os.path.exists(os.path.join(args.outputPath, os.path.dirname(filepath))):
                    os.makedirs(os.path.join(args.outputPath, os.path.dirname(filepath)))
                assert isinstance(img, PIL.Image.Image)
                img.convert("RGB").save(os.path.join(args.outputPath, filepath+".jpg"))
            except Exception as e:
                print("\nskip", e)
