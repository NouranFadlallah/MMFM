"""Section A "reported" column: counts as stated by each dataset's source,
transcribed from docs/breast_cancer_datasets.md (which cites the primary paper /
host page). None = not stated there; fill in from the paper rather than guessing.
"""
REPORTED = {
    'busbra': {'images': 1875, 'patients': 1064, 'benign': None, 'malignant': None, 'normal': 0,
               'pathology_confirmed': 'Yes (biopsy-proven)', 'license': 'CC BY 4.0'},
    'busi': {'images': 780, 'patients': 600, 'benign': 437, 'malignant': 210, 'normal': 133,
             'pathology_confirmed': 'Not stated', 'license': 'not formally stated'},
    'busc': {'images': None, 'patients': None, 'benign': None, 'malignant': None, 'normal': None,
             'pathology_confirmed': None, 'license': None},
    'breast': {'images': 256, 'patients': 256, 'benign': 154, 'malignant': 98, 'normal': 4,
               'pathology_confirmed': 'Yes (histopathology)', 'license': 'CC BY 4.0'},
    'mias': {'images': 322, 'patients': 161, 'benign': 64, 'malignant': 51, 'normal': 207,
             'pathology_confirmed': 'Not stated', 'license': 'CC BY (repository metadata)'},
    'cdd_cesm': {'images': 2006, 'patients': 326, 'benign': None, 'malignant': None, 'normal': 751,
                 'pathology_confirmed': 'Not stated', 'license': 'CC BY 4.0'},
    'cmmd': {'images': 5202, 'patients': 1775, 'benign': None, 'malignant': None, 'normal': None,
             'pathology_confirmed': 'Yes (biopsy-confirmed)', 'license': 'CC BY 4.0'},
    'bcsdbt': {'images': None, 'patients': None, 'benign': None, 'malignant': None, 'normal': None,
               'pathology_confirmed': 'Benign/Cancer biopsy-confirmed', 'license': 'CC BY-NC 4.0',
               'note': 'full collection: 5,060 participants / 5,610 studies; local copy is the validation release only'},
    'breamdm': {'images': None, 'patients': 232, 'benign': 85, 'malignant': 147, 'normal': 0,
                'pathology_confirmed': 'Not stated', 'license': 'UNVERIFIED'},
    'ispy2': {'images': None, 'patients': 982, 'benign': 0, 'malignant': 982, 'normal': 0,
              'pathology_confirmed': 'pCR outcome for a subset; diagnosis not stated',
              'license': 'CC BY-NC-ND 4.0 (BreastDCEDL)'},
}
