from setuptools import setup

setup(
   name='AI_Laptop_Py',
   version='1.0',
   description='A useful module',
   author='Man Py',
   author_email='vanhau293@gmail.com',
   packages=['AI_Laptop_Py'],  #same as name
   install_requires=['joblib', 'numpy', 'pandas','fastapi', 'pydantic', 'uvicorn'], #external packages as dependencies
)
