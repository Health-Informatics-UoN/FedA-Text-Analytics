from pathlib import Path
import pandas as pd
from functools import lru_cache

def build_snomed_filter(rel_file_path):
    """
    rel_file_path: path to sct2_Relationship_*.csv (UK Edition)
    Returns: a filter function expecting your dictionary
    """
    # Load the RF2 relationships (Is-a = typeId == 116680003)
    rels = pd.read_csv(rel_file_path, dtype=str, sep="\t")
    isa_rels = rels[rels["typeId"] == "116680003"][["sourceId", "destinationId"]]

    # Map child → parent
    parents = isa_rels

    # SNOMED top-level roots we allow
    allowed_roots = {
        "439401001",  # diagnosis
        "71388002",   # procedure
        "185361000000102",  # medication
    }

    @lru_cache(maxsize=None)
    def is_descendant(code):
        stack = [code]
        visited = set()

        while stack:
            c = stack.pop()
            if c in visited:
                continue
            visited.add(c)

            direct_parents = parents.loc[parents["sourceId"] == c, "destinationId"].tolist()

            # If any parent is a desired top-level group → accept
            if any(p in allowed_roots for p in direct_parents):
                return True

            stack.extend(direct_parents)

        return False

    def filter_annotations(my_dict):
        """
        my_dict["annotations"] → list of objects
        each annotation has annotation["label_id"] (SNOMED code)
        Returns filtered list.
        """
        result = my_dict.copy()
        result["annotations"] = []
        for annot in my_dict.get("annotations", []):
            code = str(annot.get("label_id"))
            if is_descendant(code):
                result["annotations"].append(annot)
        return result

    return filter_annotations

# Build the filter once
print("Loading SNOMED filter...")
snomed_filter = build_snomed_filter(
    "OntologyData/uk_sct2cl_41.2.0_20251119000001Z/SnomedCT_UKClinicalRF2_PRODUCTION_20251119T000001Z/Full/Terminology/sct2_Relationship_UKCLFull_GB1000000_20251119.txt"
)
print("SNOMED filter loaded.")