

with open('output.log', 'r', encoding='utf-8', errors='ignore') as rf:
    text = []
    for line in rf.readlines():
        if (
            not line.startswith('Epoch') 
            and not line.strip() == ""
            and not line.startswith('Getting')
            and not line.startswith('  0%')
            and not line.startswith('ProSST inference')
            and not line.startswith('Tokenizing')
            and not "\x1b[A" in line
        ):
            text.append(line.strip())  # \x1b is ESC)
with open('output.log', 'w') as wf:
    for t in text:
        print(t)
        wf.write(f"{t}\n")

