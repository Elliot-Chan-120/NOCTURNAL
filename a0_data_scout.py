import pandas as pd
from chembl_webresource_client.new_client import new_client
import requests
import yaml
from pathlib import Path
from b01_utility import *

def data_scout(target_input, size):
    """
    iterate through all targets and find out how many bioactivity data units we'll have to work with for each one
    \n it's better to choose an index containing a higher number since the model may not learn well from just 4-10 data points assuming those compounds make it through the filtering

    includes these two pieces of info per index as a quality assessment
    target_type: what kind of biological entity the target is
    organism: what species the target belongs to
    """
    with open("0_config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    try:
        target = new_client.target
        target_query = target.search(target_input)
        targets = pd.DataFrame.from_dict(target_query)

        if not target_query:
            raise ChEMBLAPIError(f"No targets found for query: {target_input}")
    except requests.exceptions.RequestException as e:
        raise ChEMBLAPIError(f"Network Error when attempting to connect to ChEMBL API: {e}")
    except Exception as e2:
        raise ChEMBLAPIError(f"Unexpected Error during query for target: {e2}")

    bioact_list = []

    for indx in range(len(targets)):
        selected_target = targets.target_chembl_id[indx]
        data_pts = len(new_client.activity.filter(target_chembl_id=selected_target).filter(standard_type = "IC50"))
        t_type = targets.target_type[indx]
        t_organism = targets.organism[indx]
        bioact_list.append((indx, selected_target, data_pts, t_type, t_organism))

    bioact_list.sort(key=lambda x: x[2], reverse=True)

    if cfg["data_scout_csv"] is True:
        path = Path('database') / "Target_Data"
        path.mkdir(exist_ok=True)
        scout_df = pd.DataFrame(bioact_list[:size],
                                columns=[f"{target_input}_Index", "Selected_Target", "IC50_Entries", "Target_Type", "Organism"])
        scout_df.to_csv(path / f"{target_input}_targets.csv",
                        index=False)

    for indx, selected_target, data_pts, t_type, t_organism in bioact_list[:size]:
        print(f"INDEX[{indx}] {selected_target} | {t_organism} | {t_type} | {data_pts} IC50 entries")

    return bioact_list[:size]
