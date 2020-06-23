import os

cmds = ["python train.py --dataset=data_15_12.json",
        "python train.py --dataset=data_18_9.json",
        "python train.py --dataset=data_21_6.json"
]

if __name__ == "__main__":
    for cmd in cmds:
        print("TRAIN MODEL USING: " + cmd)
        os.system(cmd)
