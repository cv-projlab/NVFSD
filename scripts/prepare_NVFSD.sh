str_="

WARNING: Building NVFSD Dataset. This process may take several hours to finish!

"
echo "$str_" && notify-send "$str_"


# GENERATE DATA
bash scripts/prepare_NVFSD_data.sh

# GENERATE LISTS
bash scripts/prepare_NVFSD_lists.sh