import json
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--n_views", type=int, default=3)
parser.add_argument("--suffix", type=str, default="")
args = parser.parse_args()

dtu_data_list = [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]
llff_data_list = ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"]
blender_data_list = ["chair", "drums", "ficus", "hotdog", "lego", "materials", "mic", "ship"]



if args.dataset_name == "LLFF":
    result_path_all = [f"output/LLFF/{scan}_{args.n_views}views_{args.suffix}/results.json" for scan in llff_data_list]
elif args.dataset_name == "DTU":
    result_path_all = [f"output/DTU/scan{scan}_{args.n_views}views_{args.suffix}/results.json" for scan in dtu_data_list]
elif args.dataset_name == "Blender":
    result_path_all = [f"output/Blender/{scan}_{args.n_views}views_{args.suffix}/results.json" for scan in blender_data_list]
else:
    raise NotImplementedError(args.dataset_name)


result = {}
for path in result_path_all:
    print(path)
    with open(path) as f:
        result_dict = json.load(f)


    for key in result_dict.keys():
        if key not in result.keys():
            result[key] = {}
        for metric_key in result_dict[key].keys():
            if metric_key not in result[key].keys():
                result[key][metric_key] = []

    for key in result_dict.keys():
        for metric_key, value in result_dict[key].items():
            result[key][metric_key].append(value)


for key in result.keys():
    for metric_key, value in result[key].items():
        print(key, metric_key, value)