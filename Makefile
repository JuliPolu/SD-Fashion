.PHONY: *

install_dependencies:
	sudo apt install python3-pip
	pip install .
	pip install -r 0fashion/requirements.txt
	export PATH=$PATH:/home/ubuntu/.local/lib/python3.10/site-packages
	wandb login

	chmod +x 0fashion/run.sh
	chmod +x 0fashion/run_sdxl_lora.sh

run_experiments:
	chmod +x 0fashion/run_sd_lora.sh
	./0fashion/run_sd_lora.sh
