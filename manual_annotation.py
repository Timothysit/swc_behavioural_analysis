import pandas as pd

### read in manual annotation files
man_annot_df = pd.read_csv('annot_manual_final.csv', sep=',')
man_stats_df = pd.read_csv('data\Annot_Manual_vs3_STATS.csv', sep=',')

