import json

with open('data/views/view_idx_list-2-3.json', 'r') as f:
    v1 = json.load(f)

with open('data/evaluation_index_re10k.json', 'r') as f:
    v2 = json.load(f)

samples = list(v2.keys())[:100]
for key in samples:
    print('origin:', v2[key])
    print('2-3:', v1[key])