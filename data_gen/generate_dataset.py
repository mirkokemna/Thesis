import os
import subprocess
from gen_uniform import generate_case

N = 10000

for casenum in range(N):
    solved = 0
    i = 0
    while solved == 0:
        i = i+1
        casetag = generate_case(casenum)
        print('Solving ' +casetag)
        print('attempt %i' %(i))
        subprocess.run(["./single_study.sh",casetag])
        maxstep = 0
        files = os.listdir(casetag)
        for file in files:
            if file.isnumeric():
                maxstep = max(maxstep,int(file))
        if maxstep % 100 != 0:
            solved = 1
    
    os.system("python3 field2image_uniform.py " + str(casenum))
