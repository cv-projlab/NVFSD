# GENERATE LISTS
userno=40

python create_lists.py NVFSD create_base_list --summary
python create_lists.py NVFSD create_list --userno "$userno" --combination A1B1C123 --gen_config --summary
python create_lists.py NVFSD create_list --userno "$userno" --combination A2B2C45 --gen_config --summary
python create_lists.py NVFSD create_list --userno "$userno" --combination A --gen_config --summary
python create_lists.py NVFSD create_list --userno "$userno" --combination A1 --gen_config --summary
python create_lists.py NVFSD create_list --userno "$userno" --combination A2 --gen_config --summary
python create_lists.py NVFSD create_list --userno "$userno" --combination B --gen_config --summary
python create_lists.py NVFSD create_list --userno "$userno" --combination B1 --gen_config --summary
python create_lists.py NVFSD create_list --userno "$userno" --combination B2 --gen_config --summary
python create_lists.py NVFSD create_list --userno "$userno" --combination C123 --gen_config --summary
python create_lists.py NVFSD create_list --userno "$userno" --combination C45 --gen_config --summary