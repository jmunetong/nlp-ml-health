bold=$(tput bold)
purple=${bold}$(tput setaf 5)
red=${bold}$(tput setaf 1)
white=${bold}$(tput setaf 7)
reset=$(tput sgr0)

trap interrupt SIGINT
function interrupt() {
    echo $red"EXECUTION TERMINATED"$reset
    exit 
}

function linebreak() {
    count=$(tput cols)
    for i in `eval echo {1..$count}`
        do 
            echo -n $white'-' 
        done
    echo $reset
}

no=$white'FALSE'$reset
yes=$white'TRUE'$reset 
model=$purple'MODEL'$reset
mask=$purple'MASK'$reset 
proc=$purple'PREPROCESS'$reset
dimension=$purple'DIMENSION'$reset

declare -a Model=('nnc_1' 'nnc_2' 'nnc_3')


for modeler in ${Model[@]}
do
    #export CUDA_VISIBLE_DEVICES=0
    echo $model'='$white${modeler} 
    python3 main.py  --model $modeler

done
