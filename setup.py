from setuptools import setup
setup(name='Penny',
      version='0.1',
      description='some help with gadget, depend on pygadgetreader',
      author='mt',
      author_email='matas.tartenas@gmail.com',
      packages=['Penny'],
      scripts=['example_scripts/density_plotter.py'],
      package_data={'Penny': ['Data/snapshot_000','Data/snapshot_500']},
)