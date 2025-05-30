mkdir -p /root/.ssh
cd /root/.ssh
wget http://s3-west.nrp-nautilus.io/mgpusim/ssh/id_rsa
wget http://s3-west.nrp-nautilus.io/mgpusim/ssh/id_rsa.pub
chmod 600 /root/.ssh/id_rsa
mkdir tmr
cd tmr
git clone https://github.com/DDantalion/tm.git
cd tm
cd bleed
chmod +x runr.sh
./runr.sh
cd ..
cd ..
cd ..
mkdir tmc
cd tmc
git clone https://github.com/DDantalion/tm.git
cd tm
cd round
chmod +x schedulec.sh
./schedulec.sh
cd ..
cd ..
cd ..
mkdir tmo
cd tmo
git clone https://github.com/DDantalion/tm.git
cd tm
cd round
chmod +x scheduleco.sh
./scheduleco.sh
cd ..
cd ..
cd ..
mkdir tml
cd tml
git clone https://github.com/DDantalion/tm.git
cd tm
cd round
chmod +x schedulecl.sh
./schedulecl.sh

mkdir tms
cd tms
git clone https://github.com/DDantalion/tm.git
cd tm
cd round
chmod +x test_migration.sh
./test_migration.sh
cd ..
cd ..
cd ..
git clone https://github.com/DDantalion/tm.git
cd tm
cd final
chmod +x final_multi.sh
./final_multi.sh

git clone https://github.com/DDantalion/tm.git
cd tm
cd cupti
nvcc cupti.cpp -o list_metrics -lcupti -lcuda
sudo ./list_metrics
chmod +x set.sh
./set.sh

git clone https://github.com/DDantalion/tm.git
cd tm
cd final
chmod +x p2pb.sh
./p2pb.sh

sudo apt install libboost-program-options-dev
git clone https://github.com/NVIDIA/nvbandwidth.git
cd nvbandwidth
cmake .
make
./nvbandwidth


# sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# sudo add-apt-repository "deb https://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture)/ /"
# sudo apt install nsight-systems
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_3/NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb
sudo apt install ./NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb 
#sudo nsys profile --gpu-metrics-devices=all --output=nvlink_report ./p2pb.sh
#nsys stats nvlink_report.nsys-rep --report list
#nvidia-smi nvlink -gt d -i 0

git clone https://github.com/pytorch/vision.git
cd vision/references/detection
pip install pycocotools
python -m torch.distributed.launch --nproc_per_node=8 --use_env \
        train.py --world-size 8 --data-path=/home/ubuntu/datasets/coco --epochs 1


mkdir -p ~/datasets/coco
cd ~/datasets/coco
# 2017 train/val images and annotations
wget http://images.cocodataset.org/zips/train2017.zip      -O train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip        -O val2017.zip
#wget http://images.cocodataset.org/zips/test2017.zip       -O test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
                                                           -O annotations_trainval2017.zip
unzip train2017.zip    && rm train2017.zip
unzip val2017.zip      && rm val2017.zip
#unzip test2017.zip     && rm test2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip



#scp -P 42144 -r root@connect.bjc1.seetacloud.com:/root/tm ./Desktop/ncontention/bleed/A100/
#scp -r ubuntu@64.181.223.194:/home/ubuntu/tmr/tm/bleed.tar.gz ./Desktop/ncontention/bleed/A100II/runr.tar.gz
#scp -r ubuntu@170.9.245.205:/home/ubuntu/tms/tm/migration.tar.gz .tar.gz ./Desktop/finalc/A100/
scp -r ubuntu@207.211.166.171:/home/ubuntu/tm/p2pb.tar.gz ./Desktop/finalc/A100/

#scp -r ubuntu@104.171.203.90:/home/ubuntu/tm/finalmulti.tar.gz ./Desktop/finalc/V100/
#scp -r ubuntu@64.181.214.142:/home/ubuntu/tml/tm/schedulec.tar.gz ./Desktop/ncontention/bleed/A100IV/schedulecl.tar.gz
#scp -r ubuntu@64.181.223.194:/home/ubuntu/tmt/tm/schedulet.tar.gz ./Desktop/ncontention/bleed/A100II/
git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests
make
./build/all_reduce_perf -g 8