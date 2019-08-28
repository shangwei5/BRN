
# R100H
python test_BRN.py   --logdir  logs/BRN/R100H   --data_path  /media/r/dataset/rain/Rain100H    --save_path result/BRN/R100H/output   --save_path_r result/BRN/R100H/rainstreak

#R100L
python test_BRN.py   --logdir  logs/BRN/R100L   --data_path  /media/r/dataset/rain/Rain100L    --save_path result/BRN/R100L/output   --save_path_r result/BRN/R100L/rainstreak

#R12
python test_BRN.py   --logdir  logs/BRN/R100L   --data_path  /media/r/dataset/rain/test12    --save_path result/BRN/R12/output   --save_path_r result/BRN/R12/rainstreak

#real
python test_real.py    --data_path  /media/r/dataset/rain/real    --save_path result/BRN/real/output   --save_path_r result/BRN/real/rainstreak
