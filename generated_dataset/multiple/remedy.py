import json
with open('generated_dataset/multiple/object_split_dic.json','r') as f:
    obj = json.load(f)

static = []
dynamic = []
both = []

for k,v in obj.items():
    if v == 'static':
        static.append(k)
    elif v == 'dynamic':
        dynamic.append(k)
    else:
        both.append(k)
        
with open('generated_dataset/multiple/split_sets.json','w') as f:
    json.dump({'static':static,'dynamic':dynamic,'both':both},f)