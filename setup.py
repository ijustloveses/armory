from setuptools import setup, find_packages


setup(name='armory',
      version='0.1.0',
      description='armory - a kaggle machine learning toolkit',
      author='Cuiyong',
      author_email='kwoo919@hotmail.com',
      package_dir={'': 'src'},
      packages=find_packages('src'),
      package_data={
              '': ['*.txt', '*.dat'],
      }
)
