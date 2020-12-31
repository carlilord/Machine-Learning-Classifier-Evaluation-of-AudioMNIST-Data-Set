import os
from glob import glob
results = [y for x in os.walk('.') for y in glob(os.path.join(x[0], '*.ipynb'))]
results = list(filter(lambda x: "checkpoint" not in x, results))
results.sort(key=lambda x: "SVM" in x)
for result in results:
    print("Calculating for {0}".format(result))
    os.system('jupyter nbconvert --to notebook --execute {0}'.format(result))