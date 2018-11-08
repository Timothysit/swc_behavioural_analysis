import pandas as pd
import numpy as np

### read in manual annotation files
man_annot_df = pd.read_csv('annot_manual_final.csv', sep=',')
print(man_annot_df)

beh_occurrence = np.zeros(8, 740)

for index, row in df.iterrows():
   if row['Code']== "Nose2Body":
       beh_occurrence[1, row['timeON']:row['timeOFF']]=1
print(beh_occurrence)



