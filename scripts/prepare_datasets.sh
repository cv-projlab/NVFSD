str_="

WARNING: Building both Datasets. This process may take several hours to finish!

"
echo "$str_" && notify-send "$str_"

bash scripts/prepare_SynFED.sh
bash scripts/prepare_NVFSD.sh