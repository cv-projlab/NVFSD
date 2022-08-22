set -e

# config_files=("configs/*.yml")

# for config_file in $config_files
# do
#   echo "Training cfg file: $config_file"
#   python runner.py --config "$config_file" -t --notif
# done

##TENSOBOARD REGEX FILTER: model*(?!.*(hparams|copy))

python runner.py --cfg configs/config-0.yml configs/base_timesformer.yml -t --notif
python runner.py --cfg configs/config-1.yml configs/base_timesformer.yml -t --notif
python runner.py --cfg configs/config-2.yml configs/base_timesformer.yml -t --notif
python runner.py --cfg configs/config-3.yml configs/base_timesformer.yml -t --notif
python runner.py --cfg configs/config-4.yml configs/base_timesformer.yml -t --notif
python runner.py --cfg configs/config-5.yml configs/base_timesformer.yml -t --notif

python runner.py --cfg configs/config-0.yml -t --notif
python runner.py --cfg configs/config-1.yml -t --notif
python runner.py --cfg configs/config-2.yml -t --notif
python runner.py --cfg configs/config-3.yml -t --notif
python runner.py --cfg configs/config-4.yml -t --notif
python runner.py --cfg configs/config-5.yml -t --notif

# python runner.py --cfg configs/config-6.yml -t --notif
# python runner.py --cfg configs/config-7.yml -t --notif
# python runner.py --cfg configs/config-8.yml -t --notif
# python runner.py --cfg configs/config-9.yml -t --notif

