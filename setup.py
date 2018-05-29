import setuptools

with open('README.md', 'r') as f:
	long_description = f.read()

setuptools.setup(
	name='corextopic',
	version='1.0.2',
	author='Greg Ver Steeg/Ryan J. Gallagher',
	author_email='gregv@isi.edu',
	keywords=['topic model', 'corex', 'anchored corex', 'LDA', 'semi-supervised', 'hierarchical topic model', 'information theory'],
	description='Hierarchical and semi-supervised topic modeling with minimal domain knowledge through Anchored Correlation Explanation',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/gregversteeg/corex_topic',
	packages=setuptools.find_packages(),
	classifiers=(
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: Apache Software License',
		'Operating System :: OS Independent',
	),
)
