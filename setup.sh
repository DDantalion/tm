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
chmod +x set.sh
./set.sh

git clone https://github.com/DDantalion/tm.git
cd tm
cd final
chmod +x p2pb.sh
./p2pb.sh

#scp -P 42144 -r root@connect.bjc1.seetacloud.com:/root/tm ./Desktop/ncontention/bleed/A100/
#scp -r ubuntu@64.181.223.194:/home/ubuntu/tmr/tm/bleed.tar.gz ./Desktop/ncontention/bleed/A100II/runr.tar.gz
#scp -r ubuntu@170.9.245.205:/home/ubuntu/tms/tm/migration.tar.gz .tar.gz ./Desktop/finalc/A100/
scp -r ubuntu@170.9.245.205:/home/ubuntu/tm/p2pb.tar.gz ./Desktop/finalc/A100/

#scp -r ubuntu@104.171.203.90:/home/ubuntu/tm/finalmulti.tar.gz ./Desktop/finalc/V100/
#scp -r ubuntu@64.181.214.142:/home/ubuntu/tml/tm/schedulec.tar.gz ./Desktop/ncontention/bleed/A100IV/schedulecl.tar.gz
#scp -r ubuntu@64.181.223.194:/home/ubuntu/tmt/tm/schedulet.tar.gz ./Desktop/ncontention/bleed/A100II/
# git clone https://github.com/NVIDIA/nccl-tests.git
# ./build/all_reduce_perf -g 8