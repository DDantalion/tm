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
mkdir tmcr
cd tmcr
git clone https://github.com/DDantalion/tm.git
cd tm
cd round
chmod +x schedulecr.sh
./schedulecr.sh
cd ..
cd ..
cd ..
mkdir tmt
cd tmt
git clone https://github.com/DDantalion/tm.git
cd tm
cd round
chmod +x schedulet.sh
./schedulet.sh
#scp -P 42144 -r root@connect.bjc1.seetacloud.com:/root/tm ./Desktop/ncontention/bleed/A100/
#scp -r ubuntu@129.146.46.231:/home/ubuntu/tmr/tm/bleed.tar.gz ./Desktop/ncontention/bleed/A100/runr.tar.gz
#scp -r ubuntu@129.146.46.231:/home/ubuntu/tmc/tm/schedulec.tar.gz ./Desktop/ncontention/bleed/A100/
#scp -r ubuntu@129.146.46.231:/home/ubuntu/tmcr/tm/schedulecr.tar.gz ./Desktop/ncontention/bleed/A100/
#scp -r ubuntu@129.146.46.231:/home/ubuntu/tmt/tm/schedulet.tar.gz ./Desktop/ncontention/bleed/A100/