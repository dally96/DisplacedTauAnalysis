with open("root_files.txt") as f:
    lines = [line.strip() for line in f if line.strip()]

prefix = 'root://cmseos.fnal.gov/'
paths = []
base_path = 'str'
i = 0
while i < len(lines):
    base_path = lines[i]
    for l in lines:
        if '.root' in l:
            base_path = lines[i] + l
            paths.append(base_path)
        else:
            
    i += 1
print(paths)
