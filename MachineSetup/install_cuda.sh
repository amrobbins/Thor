sudo apt-get update
sudo apt-get install build-essential -y
sudo apt-get install linux-headers-$(uname -r) -y
sudo apt-key del 7fa2af80
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit -y
sudo echo 'export PATH="$PATH:/usr/local/cuda/bin"' | sudo tee -a /etc/profile > /dev/null
echo "do sudo reboot now"
