# bdh-hw3-pyspark

## refer bdh-hw2-pyspark for help with pyspark and nosetests

**Work Directory: bhd-hw3-pyspark/code**

change your working directory to **code** folder after unzip, otherwise, it will raise path errors like `FileNotFound` or `ModuleNotFound`.

* **Commands to run/test code will be:**

1.  Run Each Code (```__main__``` function in each .py file): 

```
python src/main/feature_construction.py
```

2. Run all nosetests (first line) or single a single test in the test file in `src/test/`:
```
nosetests src/test/test_feature_construction.py --nologcapture
nosetests src/test/<filename>:<test_method> --nologcapture
```

* **Relative Path for imports or File-IO will be:**

1. Imports within `src/main/` forlder, e.g., import function/class from `src/main/models.py` into `src/main/feature_construction.py`, at head of `src/main/feature_construction.py`
```
import sys
sys.path.append('./')
from src.main.models import Diagnostic, Medication, LabResult
from src.main.loadRddRawData import *
```

2. Imports between from `src/main/` into `src/test/`, at head of `src/main/feature_construction.py`
```
import sys
sys.path.append('./')
from src.main.models import Diagnostic, Medication, LabResult
from src.main.feature_construction import *
from src.main.loadRddRawData import *
```

3. Read CSV files in `code/data` in `src/main/loadRddRawData.py:load_rdd_raw_data(spark)`:
```
relative_path = "./data/"           ## this is the relative path to find .csv files in code/data
file_name = ["encounter_INPUT.csv",\
            "encounter_dx_INPUT.csv", \
            "lab_results_INPUT.csv", \
             "medication_orders_INPUT.csv"]

csv_files = [relative_path+x for x in file_name]

for file in csv_files:
    df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file)
    table_name = file.split("/")[-1].split(".")[0]
    df.createOrReplaceTempView(table_name)
```
