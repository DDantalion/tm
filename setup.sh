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
chmod +x runb.sh
./runb.sh
cd ..
cd ..
cd ..
mkdir tmb
cd tmb
git clone https://github.com/DDantalion/tm.git
cd tm
cd bleed
chmod +x run.sh
./run.sh
#scp -P 42144 -r root@connect.bjc1.seetacloud.com:/root/tm ./Desktop/ncontention/bleed/A100/
#scp -r ubuntu@129.146.46.231:/home/ubuntu/tmr/tm/bleed.tar.gz ./Desktop/ncontention/bleed/A100/nbleed.tar.gz
#scp -r ubuntu@129.146.46.231:/home/ubuntu/tmb/tm/bleed.tar.gz ./Desktop/ncontention/bleed/A100/oribleed.tar.gz