"""
重複ファイルを削除するスクリプト
inputの中にあるファイルと同名のファイルをtargetから削除する
"""
import argparse as ap
import os
import shutil


# root_dir下のファイルパスを全列挙
def walk(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            yield os.path.join(root, file)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("input", help="重複元ファイルが入っているディレクトリ")
    parser.add_argument("target", help="削除する重複ファイルが入っているディレクトリ")
    parser.usage = parser.format_usage() + "重複ファイルを削除するスクリプト。inputの中にあるファイルと同名のファイルをtargetから削除する"
    args = parser.parse_args()
    input = args.input
    target = args.target
    for path in [os.path.relpath(path, input) for path in walk(input)]:
        cand_path = os.path.join(target, path)
        if os.path.exists(cand_path):
            print("delete!:", path)
            shutil.os.remove(cand_path)
