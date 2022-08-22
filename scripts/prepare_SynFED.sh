str_="

WARNING: Building SynFED Dataset. This process may take several hours to finish!

"
echo "$str_" && notify-send "$str_"

# GENERATE DATA
bash scripts/prepare_SynFED_data.sh

# GENERATE LISTS
bash scripts/prepare_SynFED_lists.sh
