# This code is based on the DRAEM project (source: https://github.com/VitjanZ/DRAEM).
# We modified and added the necessary modules or functions for our purposes.
#!/usr/bin/env bash
project_dir=$( pwd )

cd ..

mkdir datasets
cd datasets

mkdir plain
cd plain

# Download describable textures dataset
wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz
tar -xf dtd-r1.0.1.tar.gz
rm dtd-r1.0.1.tar.gz

mkdir mvtec
cd mvtec

# Download MVTec anomaly detection dataset
wget https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz
tar -xf mvtec_anomaly_detection.tar.xz
rm mvtec_anomaly_detection.tar.xz

cd ./../
mkdir btad
cd btad

# Download Bean-Tech anomaly detection dataset
wget http://avires.dimi.uniud.it/papers/btad/btad.zip
unzip -o btad.zip 
mv ./BTech_Dataset_transformed/* ./
rm -r BTech_Dataset_transformed
rm btad.zip

# generate synthetic defect data for inference 
cd $project_dir
python3 save_aug_set.py --dataset mvtec
python3 save_aug_set.py --dataset btad
