set -e

# GENERATE LISTS
userno=30
frametype=events_aets40

## Create main list
python create_lists.py SynFED create_base_list --frametype "$frametype"  --summary

## Both
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 2 --gen_cfg --summary
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 5 --gen_cfg --summary
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 10 --gen_cfg --summary
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 15 --gen_cfg --summary
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 20 --gen_cfg --summary
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 29 --gen_cfg --summary

## Only fakes
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 2 --gen_cfg --summary --no-real
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 5 --gen_cfg --summary --no-real
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 10 --gen_cfg --summary --no-real
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 15 --gen_cfg --summary --no-real
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 20 --gen_cfg --summary --no-real
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --fakeno 29 --gen_cfg --summary --no-real

## Only real
python create_lists.py SynFED create_list --frametype "$frametype" --userno "$userno" --gen_cfg --summary --no-fake

