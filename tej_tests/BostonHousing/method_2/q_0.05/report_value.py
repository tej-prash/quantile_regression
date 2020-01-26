import pandas as pd 
import numpy as np

df=pd.read_csv("tabulate_loss.csv")
print(df.index,df.columns)
print(df.head())
adaptive_lr_mean_loss=df['adaptive_lr_loss'].mean()
adaptive_lr_mean_val_loss=df['adaptive_lr_val_loss'].mean()

adaptive_lr_std_loss=df['adaptive_lr_loss'].std()
adaptive_lr_std_val_loss=df['adaptive_lr_val_loss'].std()


constant_lr_mean_loss=df['constant_lr_loss'].mean()
constant_lr_mean_val_loss=df['constant_lr_val_loss'].mean()

constant_lr_std_loss=df['constant_lr_loss'].std()
constant_lr_std_val_loss=df['constant_lr_val_loss'].std()

print("Constant LR loss",constant_lr_mean_loss,"+-", constant_lr_std_loss)
print("Constant LR val loss",constant_lr_mean_val_loss,"+-", constant_lr_std_val_loss)

print("Adaptive LR loss",adaptive_lr_mean_loss,"+-", adaptive_lr_std_loss)
print("Adaptive LR val loss",adaptive_lr_mean_val_loss,"+-", adaptive_lr_std_val_loss)