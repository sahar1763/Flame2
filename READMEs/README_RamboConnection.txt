open cmd

command: ssh -X STAFF\username@serverip
username: saharc
serverip:
	Rambo2 - 132.68.36.227
	Rambo3 - 132.68.36.231
 
Test for gpu to identify cuda:
	python3 - << 'EOF'
	import torch
	print("torch:", torch.__version__)
	print("cuda available:", torch.cuda.is_available())
	print("gpu:", torch.cuda.get_device_name(0))
	EOF

test gpu status:
	nvidia-smi
	watch -n 1 nvidia-smi

	
running script on specific gpu:
	CUDA_VISIBLE_DEVICES= 0,1,2...  python3 script.py

check what files are on the server:
ls
ls -lh (including deta)
ls -la (inclueding hidden files)

remove specific file:
rm filename.py

remove all files from folder:
rm -rf *
remove specific folder:
rm -rf folder_name
remove folder:
rmdir folder_name

get current folder:
pwd

upload new file or zip-folder to the server:
scp path\code.py STAFF\saharc@132.68.36.227:/home/saharc/

run python file:
python3 code.py

unzip folder: 
python3 -m zipfile -e source_folder.zip destination_folder

move all files one previous folder in path:
mv * ..
if there are many files this also could work:
find . -mindepth 1 -maxdepth 1 -exec mv -t .. {} +
move file:
mv filename.py new_path

change dir path:
mv old_name new_name

edit python file:
nano code_name.py

download file:  "." This is the destination. The dot means "download it to the current folder I am in right now on my computer."
scp "STAFF\saharc@132.68.36.227:/home/saharc/filename.type" .

download folder:  "." This is the destination. The dot means "download it to the current folder I am in right now on my computer."
scp -r "STAFF\saharc@132.68.36.227:/home/saharc/folder_name" .

validate folder size:
du -sh UnifiedDataset

Training run cmd
all gpus
torchrun --nproc_per_node=8 train.py --config configTrainModel.yaml
part of gpus
CUDA_VISIBLE_DEVICES=2,4,6 torchrun --nproc_per_node=3 train.py --config configTrainModel.yaml
