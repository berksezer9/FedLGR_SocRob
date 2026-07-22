#!/bin/bash

processor=gpu
strategy=all
strategy_cl=all
base=all
model=MobileNet
aug=True
rounds=10
epochs=10
icl=2
fcl=2
coef='all'

data="Data" # Path to dataset
output="" #Provide path to store output files

# --temp_dir dropped: main_fcl.py never defined this flag (and temp_dir was
# never assigned in this script either) -- passing it hard-errored argparse
# before any of this ran (see OFFICEDB_MODIFICATIONS.md).
python main_fcl.py --strategy_fl ${strategy} --strategy_cl ${strategy_cl} --model ${model} --rounds ${rounds} --epochs ${epochs} --icl ${icl} --fcl ${fcl} --path ${data} --output ${output} --aug ${aug} --processor_type ${processor} --base ${base} --reg_coef ${coef}
