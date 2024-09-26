# Executor

## Setup project
### Requirements
* Ubuntu machine
* install [pdm](https://pdm-project.org/latest/)
* python version v3.11.*
* For ssh connection test on local machine, need to install openssh-client and openssh-server
```
sudo apt-get install openssh-client openssh-server
```

Run following command to install required tools: 
```shell
chmod +x scripts/install_executor_on_ubuntu.sh && scripts/install_executor_on_ubuntu.sh
```

### Install and Run

* Install Python dependencies
```
pdm install
```

* Add .env in the project

See the required enviroment variables in the example.env

* Run project
```
python src/executor.py
```
