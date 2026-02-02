# TRE Text Analytics Pipeline
This repository is for in-development/experimental code relating to the TRE text analytics and NLP tasks.

## Test Database Installation
If there is no Postgresql database currently installed, run the docker command located in "create_postgres_container.sh".

Edit the database installation script (mimic_4_utility_scripts/import_mimic_discharge.py) to ensure your local athena vocab file path, MIMIC-IV-Note discharge csv file path and Postgres connection details are correct.

Once executed successfully, omop_cdm should appear as a schema under the postgres DB (changeable).