# Download Humap Data
kaggle competitions download -c hubmap-organ-segmentation
unzip hubmap-organ-segmentation
mv ./hubmap-organ-segmentation ./data
mkdir data_modified
mkdir data_modified/{train,test}
mkdir data_modified/train/{images,masks}
mkdir data_modified/test/{images,masks}



# Download MRI Pretrain Data
kaggle datasets download -d mateuszbuda/lgg-mri-segmentation
rm -rf lgg-mri-segmentation
mv kaggle_3m data_mri_pretrain

mkdir saved
mkdir saved/{mri,hpa}
mkdir saved/mri/{models,tensorboard}
mkdir saved/hpa/{models,tensorboard}
