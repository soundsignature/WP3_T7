# WP3_T7
 Training and testing 


## POETRY FIRST TIME:
1. Install poetry on your base python virtual environment with "pip install poetry".
2. Write "poetry init" in order to create the project.
3. In the pyproject.toml file change the name parameter in [tool.poetry] for the name "pipeline".
4. Write "poetry config virtualenvs.in-project true". This will tell poetry to locate the .venv file that will be created in the root of the project.
5. Write "poetry install". This will create the .venv file and install all the packages and dependecies needed for your project.
6. In case you want to add more, simply write "poetry add package-name".

## POETRY AFTER FIRST CONFIGURATION:
1. Install poetry on your base python virtual environment with "pip install poetry".
2. Write "poetry config virtualenvs.in-project true". This will tell poetry to locate the .venv file that will be created in the root of the project.
3. Write "poetry install". This will create the .venv file and install all the packages and dependecies needed for your project.
4. In case you want to add dependecies and package, write "poetry add package-name".

## HOW TO RUN YOUR CODE WITH YOUR POETRY VENV?
Two ways:
1. Write "poetry shell". This will activate the venv and you will be able to navigate to your script and do "python X.py". In order to exit this shell write "exit".
2. Activate the venv manually.
