#! /usr/bin/zsh

python ./linear_model/linear_model/elastic_net.py log2_azm_mic "[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]"

for outcome in log2_cfx_mic log2_cip_mic log2_cro_mic;
do
    a=1
    b=2
    c="[$a, $b]";
    for i in {1..6};
    do
        if [[ $i -gt 1 ]];
        then
            c="$c, [$a, $b]";
        fi
        python ./linear_model/linear_model/elastic_net.py $outcome "[$c]"; 
        a=$(expr $a + 2); 
        b=$(expr $b + 2); 
    done
done