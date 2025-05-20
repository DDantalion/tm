mkdir -p /root/.ssh
cd /root/.ssh
wget http://s3-west.nrp-nautilus.io/mgpusim/ssh/id_rsa
wget http://s3-west.nrp-nautilus.io/mgpusim/ssh/id_rsa.pub
chmod 600 /root/.ssh/id_rsa
mkdir tmr
cd tmr
GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no" git clone git@github.com:DDantalion/tm.git
cd tm
cd bleed
chmod +x runb.sh
./runb.sh
#scp -P 42144 -r root@connect.bjc1.seetacloud.com:/root/tm ./Desktop/ncontention/bleed/A100/