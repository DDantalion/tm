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
#scp -P 42144 -r root@connect.bjc1.seetacloud.com:/root/tm ./Desktop/ncontention/bleed/A100/
#scp -r ubuntu@64.181.223.194:/home/ubuntu/tmr/tm/bleed.tar.gz ./Desktop/ncontention/bleed/A100II/runr.tar.gz
scp -r ubuntu@64.181.214.142:/home/ubuntu/tmo/tm/scheduleo.tar.gz ./Desktop/ncontention/bleed/A100IV/scheduleco.tar.gz
scp -r ubuntu@64.181.214.142:/home/ubuntu/tml/tm/schedulel.tar.gz ./Desktop/ncontention/bleed/A100IV/schedulecl.tar.gz
#scp -r ubuntu@64.181.223.194:/home/ubuntu/tmt/tm/schedulet.tar.gz ./Desktop/ncontention/bleed/A100II/