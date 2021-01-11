autopep8 --in-place --aggressive --aggressive -r .
git clean -fxd
python3 setup.py sdist bdist_wheel
twine upload --repository testpypi dist/*
