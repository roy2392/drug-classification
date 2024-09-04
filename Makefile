install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:	
	black *.py 

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./results/metrics.txt >> report.md
	
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./results/model_results.png)' >> report.md
	
	cml comment create report.md
		
update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin ${branch}

hf-login: 
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF)

push-hub:
    git add remote origin https://huggingface.co/spaces/roeyzalta/Drug-Classification
    git remote add ./app/app.py
    git commit -m "Sync App files"
	huggingface-cli upload roy2392/Drug-Classification ./app --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload roy2392/Drug-Classification ./model /model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload roy2392/Drug-Classification ./results /metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub

all: install format train eval update-branch deploy
