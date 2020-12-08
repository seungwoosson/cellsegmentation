#%%
import argparse

parser = argparse.ArgumentParser(description='Argparse Tutorial')
# argument는 원하는 만큼 추가한다.
parser.add_argument('--print-number', type=int, 
                    help='an integer for printing repeatably')

args = parser.parse_args()

for i in range(args.print_number):
    print('print number {}'.format(i+1))
# %%
!python 'C:\Users\user\Documents\GitHub\unet_cellseg\prog.py' --print-number 5
# %%
