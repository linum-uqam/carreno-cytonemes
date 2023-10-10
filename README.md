# Cytonemes Carreno
Carreno project for cytonemes metrics extraction via 3D volumes.
Features volume data provided by UDM collaborators and all implemented metrics extraction pipelines.  

This repository is to reunite most of the Carreno project in one place.

The library and scripts can be installed locally by using:
```
pip install -e .
```

## Dependencies
With `pip` (download cuda on nvidia website for GPU usage):
```
pip install -r requirements.txt
```

With `anaconda` :
```
conda env create --file environment.yml
conda activate carreno-cyto
```

## Project structure
Will try to respect [SCILPY](https://github.com/scilus/scilpy "SCILPY GitHub") project structure as much as possible.