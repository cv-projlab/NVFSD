set -e

# config_files=("configs/*.yml")

# for config_file in $config_files
# do
#   echo "Training cfg file: $config_file"
#   python runner.py --config "$config_file" -t --notif
# done



python runner.py --cfg configs/config-0.yml configs/base_timesformer.yml -i -p data/out/weights/model_2022-05-30-08-36-22.pth -b 2 #--notif
python runner.py --cfg configs/config-1.yml configs/base_timesformer.yml -i -p data/out/weights/model_2022-05-30-15-46-35.pth -b 2 #--notif
python runner.py --cfg configs/config-2.yml configs/base_timesformer.yml -i -p data/out/weights/model_2022-05-30-23-50-54.pth -b 8 #--notif
python runner.py --cfg configs/config-3.yml configs/base_timesformer.yml -i -p data/out/weights/model_2022-05-31-09-48-59.pth -b 8 #--notif
python runner.py --cfg configs/config-4.yml configs/base_timesformer.yml -i -p data/out/weights/model_2022-05-31-21-51-55.pth -b 8 #--notif
python runner.py --cfg configs/config-5.yml configs/base_timesformer.yml -i -p data/out/weights/model_2022-06-01-11-22-25.pth -b 8 #--notif
#8
python runner.py --cfg configs/config-0.yml -i -p data/out/weights/model_2022-06-02-04-14-41.pth -b 8 #--notif
python runner.py --cfg configs/config-1.yml -i -p data/out/weights/model_2022-06-02-05-15-18.pth -b 8 #--notif
python runner.py --cfg configs/config-2.yml -i -p data/out/weights/model_2022-06-02-06-22-44.pth -b 8 #--notif
python runner.py --cfg configs/config-3.yml -i -p data/out/weights/model_2022-06-02-07-44-18.pth -b 8 #--notif
python runner.py --cfg configs/config-4.yml -i -p data/out/weights/model_2022-06-02-09-19-02.pth -b 8 #--notif
python runner.py --cfg configs/config-5.yml -i -p data/out/weights/model_2022-06-02-11-06-12.pth -b 8 #--notif


