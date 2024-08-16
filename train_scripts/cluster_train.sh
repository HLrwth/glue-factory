#!/usr/bin/env bash
# prepare dataset
#mksquashfs ./datasets ./datasets.squashfs -noI -noD -noF -noX -no-fragments -no-duplicates

mkdir /tmp/.cache
export PYTORCH_KERNEL_CACHE_PATH=/tmp/.cache

mkdir -p /tmp/gluefactory_train/data/revisitop1m
squashfuse /is/cluster/fast/hli/glue/data/revisitop1m.squashfs /tmp/gluefactory_train/data/revisitop1m
cp -r /home/hli/glue-factory/gluefactory /tmp/gluefactory_train/
cd /tmp/gluefactory_train

source /home/hli/glue-factory/train_scripts/setCUDAenv.sh 11.8
/is/cluster/hli/miniconda3/envs/attnvo/bin/python -m gluefactory.train "$1" --conf gluefactory/configs/superpoint+simpleglue_homography.yaml

TARGET_DIR="/is/cluster/fast/hli/glue"
mkdir -p "$TARGET_DIR"
cp -r /tmp/gluefactory_train/outputs "$TARGET_DIR"