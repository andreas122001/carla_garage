#!/usr/bin/env bash

# Download and install CARLA
mkdir carla
cd carla
wget https://tiny.carla.org/carla-0-9-15-linux
tar -xf CARLA_0.9.15.tar.gz
cd Import
wget https://tiny.carla.org/additional-maps-0-9-15-linux
cd ..
./ImportAssets.sh
rm CARLA_0.9.15.tar.gz
cd ..