# run stuff

for j in {-1..2}
do
    python3 model_fitting.py -s $j -r &
done
