upload:
	make clean
	python setup.py sdist bdist_wheel && twine upload dist/*
clean:
	python setup.py clean --all
	pyclean .
	rm -rf *.pyc __pycache__ build dist gym_brain.egg-info gym_brain/__pycache__ gym_brain/units/__pycache__ tests/__pycache__ tests/reports docs/build
