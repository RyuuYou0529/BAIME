# DLite

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-black.json)](https://github.com/copier-org/copier)

A Simple Blueprint for Deep Learing Project

## Features

DLite guide users to build their deep learning project in serval modules in ***lib/***:

- **arch**: model architecture
- **dataset**: data reader
- **loss**: loss function
- **optimizer**: weights optimizer
- **scheduler**: learning rate scheduler
- **trainer**: logic of training process
- **inferencer**: logic of inferencing process
- **utils**: helper functions 

## Usage

```bash
copier copy --trust https://github.com/RyuuYou0529/DLite ${your_project_workspace}
```
