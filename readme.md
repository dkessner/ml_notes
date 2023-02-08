# ML notes


## virtual environment setup

```
MY_ENV=~/path/to/my_env
```

Create a new virtual environment
```
python3 -m venv $MY_ENV
```

Activate the virtual environment
```
source $MY_ENV/bin/activate
```

Install stuff
```
pip install numpy
```

Install required packages
```
pip install -r requirements.txt
```

Create requirements.txt
```
pip freeze > requirements.txt
```


