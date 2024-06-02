# -*- coding: utf-8 -*-
import pandas as pd

pipeline_csv = "data/output/unet3D/slik1.csv"  # pipeline results
data_txt =     "data/drawn_cyto/slik1-1.txt"  # Basile hand drawn cyto
data_img =     "data/drawn_cyto/slik1-1.tif"  # z projection img

viz_data_txt = 1  # show data_txt plot over data_img

pipeline_df = pd.read_csv(pipeline_csv)
data_df = pd.read_csv(data_txt, sep="\t")

# transform pipeline dataframe to have same format as data
id, xi, yi, xf, yf, length = data_df.columns
transformed_map = {c:[] for c in data_df.columns}
for path_id in pipeline_df["path_id"].unique():
    path = pipeline_df[pipeline_df["path_id"] == path_id]
    id, xi, yi, xf, yf, length = data_df.columns
    _, _, _, rt, pid, x1, y1, z1, _, l1 = path.iloc[0]
    _, _, _, rt, pid, x2, y2, z2, _, l2 = path.iloc[-1]
    transformed_map[id].append(path_id)
    transformed_map[xi].append(x1)
    transformed_map[yi].append(y1)
    transformed_map[xf].append(x2)
    transformed_map[yf].append(y2)
    transformed_map[length].append(l2)

transformed_df = pd.DataFrame(transformed_map)

print(len(transformed_df), "paths found with pipeline")
print(len(data_df), "paths found by hand")
print()

# match and compare paths
matches = []
used_d_path = []
for p_row in transformed_df.iloc:
    ssd = 1e9
    pid_match = None
    for i, d_row in enumerate(data_df.iloc):
        if i in used_d_path:
            continue
        xi1 = int(d_row[xi])
        yi1 = int(d_row[yi])
        xi2 = int(p_row[xi])
        yi2 = int(p_row[yi])
        xf1 = int(d_row[xf])
        yf1 = int(d_row[yf])
        xf2 = int(p_row[xf])
        yf2 = int(p_row[yf])
        distance_i = ((xi1 - xi2) ** 2 + (yi1 - yi2) ** 2) ** 0.5
        distance_f = ((xf1 - xf2) ** 2 + (yf1 - yf2) ** 2) ** 0.5
        distance = distance_i + distance_f
        if distance < ssd:
            ssd = distance
            pid_match = i # d_row[id] would have made sense if ids weren't all equal ...
    used_d_path.append(pid_match)
    matches.append([pid_match, p_row[id], ssd])

total = 0
for m in matches:
    print(m[0], " data path matches best with predicted path ", int(m[1]),
          " (ssd = ", round(m[2], 2), ")",sep="")
    total += m[2]

for i, d_row in enumerate(data_df.iloc):
    if not any(i == m[0] for m in matches):
        print("data path", i, "doesn't have any match")

print("avg diff", total/len(matches))
