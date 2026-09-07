#!/usr/bin/env python3
"""Sequentially fetch public breast-cancer datasets and track status in
docs/dataset_download_links.md.

Only datasets with a plain, unauthenticated HTTP source (Zenodo, Figshare,
a stable direct URL, or a small GitHub repo) are actually downloaded. Datasets
that require a login, a signed data-use agreement, an institutional
application, or a specialized retrieval tool (TCIA's NBIA Data Retriever,
PhysioNet credentialing, Kaggle API keys, Synapse, GDC controlled access,
Aspera/Globus transfers, etc.) are never attempted -- they are recorded as
"manual" with the reason, since this script cannot and should not try to
route around those gates.

Every attempt (success, failure, skip, or manual) updates a small JSON state
file and regenerates docs/dataset_download_links.md from it, so the table
always reflects the most recent run. Re-running is safe: already-downloaded
files are left in place and re-checked, not re-fetched.

Usage:
    python3 scripts/download_datasets.py --list
    python3 scripts/download_datasets.py --init-md
    python3 scripts/download_datasets.py --all --max-size-gb 2 --out datasets/raw
    python3 scripts/download_datasets.py --dataset oasbud --max-size-gb 5
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MD_PATH = REPO_ROOT / "docs" / "dataset_download_links.md"
STATE_PATH = REPO_ROOT / "datasets" / ".download_state.json"
DEFAULT_OUT_DIR = REPO_ROOT / "datasets" / "raw"

USER_AGENT = "MMFM-dataset-downloader/1.0 (+https://github.com/NouranFadlallah/MMFM)"


@dataclasses.dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    category: str
    info_url: str
    access: str  # "zenodo" | "figshare" | "direct" | "github_clone" | "manual"
    source: Optional[str] = None  # zenodo record id / figshare article id / direct URL / git URL
    notes: str = ""


# Order matches the download run order: MRI is deliberately last because its
# datasets are by far the largest in this inventory.
DATASETS: list[Dataset] = [
    # ---------------- Mammography ----------------
    Dataset("bcdr", "BCDR", "Mammography", "http://bcdr.ceta-ciemat.es", "manual",
            notes="Request-based access; both known hosts were unreachable during research."),
    Dataset("cbis_ddsm", "CBIS-DDSM", "Mammography",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629", "manual",
            notes="TCIA collection; requires NBIA Data Retriever."),
    Dataset("cdd_cesm", "CDD-CESM", "Mammography",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=109379611", "manual",
            notes="TCIA collection; requires Aspera/Faspex."),
    Dataset("cmmd", "CMMD", "Mammography",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230508", "manual",
            notes="TCIA collection; requires NBIA Data Retriever."),
    Dataset("csaw_cc", "CSAW-CC", "Mammography",
            "https://researchdata.se/en/catalogue/dataset/2021-204-1", "manual",
            notes="Metadata CSV is open; imaging files are available on request only."),
    Dataset("ddsm", "DDSM", "Mammography", "http://www.eng.usf.edu/cvprg/Mammography/Database.html", "manual",
            notes="Original host largely defunct; use CBIS-DDSM instead."),
    Dataset("dmid", "DMID", "Mammography",
            "https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883",
            "figshare", source="24522883", notes="Figshare article; open download."),
    Dataset("embed", "EMBED", "Mammography",
            "https://registry.opendata.aws/emory-breast-imaging-dataset-embed/", "manual",
            notes="Controlled access via request form + Research Use Agreement."),
    Dataset("inbreast", "INbreast", "Mammography",
            "http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database", "manual",
            notes="Request-based (email); official host DNS was unreachable during research."),
    Dataset("kau_bcmd", "KAU-BCMD", "Mammography",
            "https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset", "manual",
            notes="Kaggle dataset; needs a Kaggle API token to script."),
    Dataset("mammo_mx", "Mammo-MX", "Mammography", "https://zenodo.org/records/17740027",
            "zenodo", source="17740027", notes="Open Zenodo record; ~74.6 GB total."),
    Dataset("mammosightr", "MammosighTR", "Mammography", "https://pubs.rsna.org/doi/10.1148/ryai.240841", "manual",
            notes="Access terms unverified; likely request/DUA-based (national registry data)."),
    Dataset("nl_breast_screening", "NL-Breast-Screening", "Mammography",
            "https://www.frdr-dfdr.ca/repo/dataset/cb5ddb98-ccdf-455c-886c-c9750a8c34c2", "manual",
            notes="FRDR; distributed via Globus transfer, not a plain HTTP file."),
    Dataset("omidb", "OMI-DB / OPTIMAM (imaging)", "Mammography",
            "https://medphys.royalsurrey.nhs.uk/omidb/", "manual",
            notes="Application-based; Data Access Committee review required."),
    Dataset("rsna_mammo", "RSNA Screening Mammography Breast Cancer Detection", "Mammography",
            "https://www.kaggle.com/competitions/rsna-breast-cancer-detection", "manual",
            notes="Kaggle competition; needs a Kaggle API token + rules acceptance."),
    Dataset("vindr_mammo", "VinDr-Mammo", "Mammography",
            "https://physionet.org/content/vindr-mammo/1.0.0/", "manual",
            notes="PhysioNet; requires a credentialed account and signed DUA."),
    Dataset("bcs_dbt", "BCS-DBT (Duke, digital breast tomosynthesis)", "Mammography",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=64685580", "manual",
            notes="TCIA collection; requires NBIA Data Retriever."),

    # ---------------- Ultrasound ----------------
    Dataset("aln_ultra", "ALN-Ultra", "Ultrasound", "https://zenodo.org/records/15003119",
            "zenodo", source="15003119", notes="Open Zenodo record; reported total ~16.4 TB."),
    Dataset("bcmid", "BCMID", "Ultrasound", "https://zenodo.org/records/14970848", "manual",
            notes="Zenodo record but files require a logged-in Zenodo session to download."),
    Dataset("buid_qamebi", "BUID / QAMEBI", "Ultrasound",
            "https://qamebi.com/breast-ultrasound-images-database/", "manual",
            notes="Direct zip links exist on the page but weren't confirmed stable enough to script."),
    Dataset("bus_cot", "BUS-CoT", "Ultrasound", "https://doi.org/10.6084/m9.figshare.30838715",
            "figshare", source="30838715", notes="Open Figshare record."),
    Dataset("busis", "BUSIS", "Ultrasound", "https://arxiv.org/abs/1801.03182", "manual",
            notes="No working public download link found; likely requires contacting authors."),
    Dataset("bus_uc", "BUS_UC", "Ultrasound", "https://data.mendeley.com/datasets/3ksd7w7jkx/1", "manual",
            notes="Mendeley Data; download bundling isn't a stable scriptable URL."),
    Dataset("busi_whu", "BUSI_WHU", "Ultrasound", "https://data.mendeley.com/datasets/k6cpmwybk3/1", "manual",
            notes="Mendeley Data; download bundling isn't a stable scriptable URL."),
    Dataset("bus_uclm", "BUS-UCLM", "Ultrasound", "https://data.mendeley.com/datasets/7fvgj4jsp7/1", "manual",
            notes="Mendeley Data; download bundling isn't a stable scriptable URL."),
    Dataset("cadbusi", "CADBUSI", "Ultrasound", "https://datascienceuwl.github.io/CADBUSI/", "manual",
            notes="Not publicly downloadable; likely requires a DUA with Mayo Clinic."),
    Dataset("cva_net_buv", "CVA-Net / BUV (Breast Ultrasound Video)", "Ultrasound",
            "https://github.com/jhl-Det/CVA-Net", "manual",
            notes="Repo is code only; video data hosted on Google Drive/Baidu links inside it."),
    Dataset("gdph_sysucc", "GDPH&SYSUCC", "Ultrasound", "https://github.com/yuhaomo/HoVerTrans", "manual",
            notes="Repo is code only; data hosted on a OneDrive link inside it."),
    Dataset("oasbud", "OASBUD", "Ultrasound", "https://zenodo.org/records/545928",
            "zenodo", source="545928", notes="Open Zenodo record; ~296.8 MB, small."),
    Dataset("stu_hospital", "STU-Hospital dataset", "Ultrasound",
            "https://doi.org/10.1371/journal.pone.0221535", "manual",
            notes="No direct public download link found; circulates only via third-party repos."),
    Dataset("tdsc_abus2023", "TDSC-ABUS2023", "Ultrasound",
            "https://tdsc-abus2023.grand-challenge.org/Dataset/", "manual",
            notes="Requires a signed data-use agreement emailed to the organizers."),
    Dataset("udiat", "UDIAT / Dataset B", "Ultrasound",
            "https://helward.mmu.ac.uk/STAFF/m.yap/dataset.php", "manual",
            notes="Not directly downloadable; requires contacting the authors."),
    Dataset("us3m", "US3M", "Ultrasound",
            "https://www.kaggle.com/datasets/timesxy/multimodal-breast-ultrasound-dataset-us3m", "manual",
            notes="Kaggle dataset; needs a Kaggle API token to script."),

    # ---------------- CT & Less-Common Modalities ----------------
    Dataset("uc_davis_phantom", "UC Davis Breast Phantom Dataset", "CT & Other", "https://zenodo.org/records/4529852",
            "zenodo", source="4529852", notes="Open Zenodo record; ~95.7 GB."),
    Dataset("aapm_dl_spectral_ct", "AAPM DL-Spectral CT Challenge (simulated)", "CT & Other",
            "https://www.aapm.org/GrandChallenge/DL-spectral-CT/", "manual",
            notes="Challenge page; no confirmed stable direct-download URL."),
    Dataset("dmr_ir", "DMR-IR (thermography)", "CT & Other", "https://visual.ic.uff.br/dmi", "manual",
            notes="Free registration required. Unofficial mirror: huggingface.co/datasets/SemilleroCV/DMR-IR."),
    Dataset("um_bmid", "UM-BMID (microwave)", "CT & Other",
            "https://ieee-dataport.org/open-access/university-manitoba-breast-microwave-imaging-dataset-um-bmid",
            "manual", notes="IEEE DataPort; requires a free account login."),
    Dataset("mri_derived_microwave_models", "MRI-Derived Numerical Breast Models Repository", "CT & Other",
            "https://github.com/acpelicano/breast_models_repository", "github_clone",
            source="https://github.com/acpelicano/breast_models_repository.git",
            notes="Small GitHub repo; cloned directly."),

    # ---------------- Pathology / Histopathology ----------------
    Dataset("acrobat", "ACROBAT", "Pathology", "https://researchdata.se/en/catalogue/dataset/2022-190-1", "manual",
            notes="SND-hosted; access process not a plain HTTP download."),
    Dataset("aidpath", "AIDPATH", "Pathology", "https://mitel.dimi.uniud.it/aidpath-db/", "manual",
            notes="Account registration required."),
    Dataset("bach", "BACH", "Pathology", "https://zenodo.org/records/3632035",
            "zenodo", source="3632035", notes="Open Zenodo record; ~13.4 GB."),
    Dataset("bcnb", "BCNB", "Pathology", "https://bcnb.grand-challenge.org/", "manual",
            notes="Grand Challenge dataset; access process not a plain HTTP download."),
    Dataset("bracs", "BRACS", "Pathology", "https://www.bracs.icar.cnr.it/download/", "manual",
            notes="Likely requires registration; not a stable direct URL."),
    Dataset("breakhis", "BreakHis", "Pathology",
            "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz", "direct",
            source="http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz",
            notes="Direct static tar.gz."),
    Dataset("brecahad", "BreCaHAD", "Pathology", "https://doi.org/10.6084/m9.figshare.7379186",
            "figshare", source="7379186", notes="Open Figshare record."),
    Dataset("camelyon16", "CAMELYON16", "Pathology", "https://camelyon16.grand-challenge.org/Data/", "manual",
            notes="~400 WSIs via GigaScience/AWS Open Data/Baidu Pan; very large, no single stable URL."),
    Dataset("camelyon17", "CAMELYON17", "Pathology", "https://camelyon17.grand-challenge.org/Data/", "manual",
            notes="~1,000 WSIs via GigaScience/AWS Open Data/Baidu Pan; very large, no single stable URL."),
    Dataset("cptac_brca_path", "CPTAC-BRCA (pathology slides)", "Pathology",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70227748", "manual",
            notes="TCIA collection; requires NBIA Data Retriever."),
    Dataset("herohe", "HEROHE Challenge", "Pathology", "https://ecdp2020.grand-challenge.org/Dataset/", "manual",
            notes="Google Drive link gated behind reading the challenge Rules page."),
    Dataset("her2_warwick", "HER2 Challenge Contest (Warwick)", "Pathology", "https://warwick.ac.uk/tia", "manual",
            notes="TIA Centre page returned HTTP 403 during research; access unconfirmed."),
    Dataset("her2_ihc_40x", "HER2-IHC-40x", "Pathology", "https://zenodo.org/records/15179608",
            "zenodo", source="15179608", notes="Open Zenodo record."),
    Dataset("idc_patches", "IDC Histology Patches (Janowczyk/Kaggle)", "Pathology",
            "https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images", "manual",
            notes="Kaggle dataset; needs a Kaggle API token to script."),
    Dataset("impress", "IMPRESS", "Pathology", "https://github.com/huangzhii/IMPRESS", "manual",
            notes="Raw WSIs not published in the repo; contact corresponding author."),
    Dataset("mitos_2012", "MITOS (ICPR 2012 Contest)", "Pathology",
            "https://github.com/znck/mitosis-detection", "manual",
            notes="Original contest site defunct; only an unofficial, unverified mirror was found."),
    Dataset("mitos_atypia_14", "MITOS-ATYPIA-14", "Pathology",
            "https://mitos-atypia-14.grand-challenge.org/Dataset/", "manual",
            notes="Google Drive links on the challenge page; not a stable scriptable URL."),
    Dataset("patchcamelyon", "PatchCamelyon (PCam)", "Pathology", "https://zenodo.org/records/2546921",
            "zenodo", source="2546921", notes="Zenodo mirror of the GitHub-hosted dataset."),
    Dataset("post_nat_brca", "Post-NAT-BRCA", "Pathology",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758117", "manual",
            notes="TCIA collection; requires Aspera/Faspex."),
    Dataset("sln_breast", "SLN-Breast", "Pathology",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52763339", "manual",
            notes="TCIA collection; requires Aspera/Faspex, ~53 GB."),
    Dataset("tcga_brca_path", "TCGA-BRCA (pathology slides)", "Pathology",
            "https://portal.gdc.cancer.gov/projects/TCGA-BRCA", "manual",
            notes="GDC Data Portal; JS-rendered, needs the GDC Data Transfer Tool / API auth for bulk download."),
    Dataset("tiger", "TIGER Challenge", "Pathology", "https://registry.opendata.aws/tiger/", "manual",
            notes="AWS Open Data (S3); scriptable in principle via aws-cli/boto3, out of scope for this script."),
    Dataset("tupac16", "TUPAC16", "Pathology", "https://tupac.grand-challenge.org/Dataset/", "manual",
            notes="Google Drive links on the challenge page; not a stable scriptable URL."),
    Dataset("bahcesehir_path", "Digital Pathology Dataset for Breast Cancer Diagnosis (Bahcesehir Univ.)",
            "Pathology", "https://zenodo.org/records/14131968",
            "zenodo", source="14131968", notes="Open Zenodo record; reported ~11.2 TB, very large."),
    Dataset("hires_path_nac", "High-Resolution Digital Pathology Imaging of Breast Cancer", "Pathology",
            "https://healthdatanexus.ai/content/breastcancerimaging/1.0.0/", "manual",
            notes="Restricted: credentialed user status, ethics training, and a signed DUA required."),
    Dataset("fnac_c1_c5", "Multi-Center Breast FNAC Whole-Slide Cytology Dataset (C1-C5)", "Pathology",
            "https://arxiv.org/abs/2606.30209", "manual",
            notes="Zenodo record referenced in the preprint but its exact id wasn't confirmed."),

    # ---------------- Clinical, Genomic & Biomarker Data ----------------
    Dataset("bcsc_rf", "BCSC Risk Factor Dataset", "Clinical/Genomic",
            "https://www.bcsc-research.org/index.php/datasets/rf", "manual",
            notes="Host was unreachable during research; custom linked data needs a DUA regardless."),
    Dataset("cptac_brca_omics", "CPTAC-BRCA (proteogenomics)", "Clinical/Genomic",
            "https://pdc.cancer.gov/pdc/browse", "manual",
            notes="Proteomic Data Commons; JS-rendered portal, needs its API/CLI tooling."),
    Dataset("geo_gse2034", "GEO GSE2034", "Clinical/Genomic",
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2034", "direct",
            source="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2034/matrix/GSE2034_series_matrix.txt.gz",
            notes="Direct GEO FTP series-matrix file; small."),
    Dataset("geo_gse96058", "GSE96058 / SCAN-B", "Clinical/Genomic",
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058", "direct",
            source="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/matrix/GSE96058-GPL11154_series_matrix.txt.gz",
            notes="Direct GEO FTP series-matrix file (main GPL11154 platform; a smaller GPL18573 file also exists)."),
    Dataset("ispy2_clinical", "I-SPY2 (clinical/biomarker annotations)", "Clinical/Genomic",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230072", "manual",
            notes="TCIA; clinical CSV is bundled with the imaging collection download."),
    Dataset("metabric", "METABRIC", "Clinical/Genomic",
            "https://www.cbioportal.org/study/summary?id=brca_metabric", "manual",
            notes="cBioPortal; bulk download needs its API, raw sequence data is EGA controlled-access."),
    Dataset("msk_impact_brca", "MSK-IMPACT (breast cohort)", "Clinical/Genomic",
            "https://www.cbioportal.org/study/summary?id=breast_msk_2025", "manual",
            notes="cBioPortal; bulk download needs its API."),
    Dataset("omidb_clinical", "OPTIMAM (OMI-DB) clinical data", "Clinical/Genomic",
            "https://medphys.royalsurrey.nhs.uk/omidb/", "manual",
            notes="Application-based; Data Access Committee review required."),
    Dataset("plco_breast", "PLCO (breast dataset)", "Clinical/Genomic",
            "https://cdas.cancer.gov/datasets/plco/19/", "manual",
            notes="Requires an approved CDAS project and data transfer agreement."),
    Dataset("seer_breast", "SEER (breast cohort)", "Clinical/Genomic", "https://seer.cancer.gov/", "manual",
            notes="SEER*Stat software + registration required for most extracts."),
    Dataset("tcga_brca_clinical", "TCGA-BRCA (clinical/genomic)", "Clinical/Genomic",
            "https://portal.gdc.cancer.gov/projects/TCGA-BRCA", "manual",
            notes="GDC Data Portal; JS-rendered, needs the GDC Data Transfer Tool / API auth for bulk download."),
    Dataset("wbcd", "WBCD / WDBC (Wisconsin Diagnostic Breast Cancer Dataset)", "Clinical/Genomic",
            "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic", "direct",
            source="https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
            notes="Direct static zip from UCI; tiny (<1 MB)."),

    # ---------------- MRI (deliberately last: by far the largest datasets) ----------------
    Dataset("acrin_6698", "ACRIN-6698 / I-SPY2 Breast DWI", "MRI",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=50135447", "manual",
            notes="TCIA collection; requires NBIA Data Retriever, ~842 GB."),
    Dataset("bmmr2", "BMMR2 Challenge", "MRI",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=89096426", "manual",
            notes="TCIA collection; requires NBIA Data Retriever."),
    Dataset("breast_diagnosis", "Breast-Diagnosis (TCIA)", "MRI",
            "https://wiki.cancerimagingarchive.net/display/Public/BREAST-DIAGNOSIS", "manual",
            notes="TCIA collection; requires NBIA Data Retriever."),
    Dataset("breast_mri_nact_pilot", "Breast-MRI-NACT-Pilot", "MRI",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22513764", "manual",
            notes="TCIA collection; requires NBIA Data Retriever, ~19.5 GB."),
    Dataset("breastdcedl", "BreastDCEDL", "MRI", "https://zenodo.org/records/17274053",
            "zenodo", source="17274053", notes="Open Zenodo record; standardized 2,070-patient DCE-MRI set, large."),
    Dataset("breastdcedl_ispy2", "BreastDCEDL-ISPY2", "MRI",
            "https://www.cancerimagingarchive.net/analysis-result/breastdcedl-ispy2/", "manual",
            notes="TCIA; requires Aspera Connect, ~54 GB."),
    Dataset("duke_breast_mri", "Duke-Breast-Cancer-MRI", "MRI",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903", "manual",
            notes="TCIA collection; requires NBIA Data Retriever, ~368 GB."),
    Dataset("ea1141", "EA1141", "MRI", "https://www.cancerimagingarchive.net/collection/ea1141/", "manual",
            notes="TCIA collection; requires NBIA Data Retriever, ~2.82 TB."),
    Dataset("euro_multicenter_mri", "European Multi-Center Breast Cancer MRI Dataset", "MRI",
            "https://zenodo.org/records/15075570", "zenodo", source="15075570",
            notes="Open Zenodo record covers a ~500-subject/13GB portion; full set needs Health-RI XNAT."),
    Dataset("fastmri_breast", "fastMRI Breast", "MRI", "https://fastmri.med.nyu.edu/", "manual",
            notes="Requires signing NYU's fastMRI data-sharing agreement."),
    Dataset("ispy1", "I-SPY1 / ACRIN 6657", "MRI",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=20643859", "manual",
            notes="TCIA collection; requires NBIA Data Retriever, ~76.2 GB."),
    Dataset("ispy1_seg_radiomics", "I-SPY1-Tumor-SEG-Radiomics", "MRI",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=101942541", "manual",
            notes="TCIA; requires Aspera Connect."),
    Dataset("ispy2_mri", "I-SPY2 (imaging)", "MRI",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230072", "manual",
            notes="TCIA collection; requires NBIA Data Retriever, ~1.6 TB."),
    Dataset("mama_mia", "MAMA-MIA", "MRI", "https://www.synapse.org/Synapse:syn60868042", "manual",
            notes="Synapse; requires an account and dataset-specific access request."),
    Dataset("qin_breast_67", "QIN-Breast (67-subject Philips cohort)", "MRI",
            "https://wiki.cancerimagingarchive.net/display/Public/QIN-Breast", "manual",
            notes="TCIA collection; requires NBIA Data Retriever."),
    Dataset("qin_breast_dce_10", "QIN-BREAST-DCE-MRI (10-patient Siemens cohort)", "MRI",
            "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=18514286", "manual",
            notes="TCIA collection; requires NBIA Data Retriever."),
    Dataset("rider_breast_mri", "RIDER Breast MRI", "MRI",
            "https://wiki.cancerimagingarchive.net/display/Public/RIDER+Breast+MRI", "manual",
            notes="TCIA collection; requires NBIA Data Retriever (small, ~401 MB, but same tool-only access)."),
]


# --------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------

class SizeExceeded(Exception):
    def __init__(self, written: int):
        self.written = written


def _urlopen(url: str, method: str = "GET", timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def zenodo_files(record_id: str) -> list[tuple[str, str, Optional[int]]]:
    with _urlopen(f"https://zenodo.org/api/records/{record_id}") as r:
        data = json.load(r)
    return [(f["key"], f["links"]["self"], f.get("size")) for f in data.get("files", []) or []]


def figshare_files(article_id: str) -> list[tuple[str, str, Optional[int]]]:
    with _urlopen(f"https://api.figshare.com/v2/articles/{article_id}") as r:
        data = json.load(r)
    return [(f["name"], f["download_url"], f.get("size")) for f in data.get("files", []) or []]


def head_size(url: str) -> Optional[int]:
    try:
        with _urlopen(url, method="HEAD", timeout=30) as r:
            length = r.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download_stream(url: str, dest: Path, max_bytes: Optional[int]) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    written = 0
    with _urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if max_bytes is not None and written > max_bytes:
                f.close()
                tmp.unlink(missing_ok=True)
                raise SizeExceeded(written)
            f.write(chunk)
    tmp.rename(dest)
    return written


def resolve_files(ds: Dataset) -> list[tuple[str, str, Optional[int]]]:
    if ds.access == "zenodo":
        return zenodo_files(ds.source)
    if ds.access == "figshare":
        return figshare_files(ds.source)
    if ds.access == "direct":
        filename = ds.source.rstrip("/").split("/")[-1] or f"{ds.id}.bin"
        return [(filename, ds.source, head_size(ds.source))]
    raise ValueError(f"resolve_files unsupported for access={ds.access!r}")


def attempt_download(ds: Dataset, out_dir: Path, max_size_gb: Optional[float]) -> dict:
    max_bytes = None if max_size_gb is None else int(max_size_gb * (1024 ** 3))

    if ds.access == "manual":
        return {"status": "manual", "detail": ds.notes or "Requires manual/gated access."}

    if ds.access == "github_clone":
        ds_dir = out_dir / ds.id
        if ds_dir.exists() and any(ds_dir.iterdir()):
            return {"status": "success", "detail": f"already present at {ds_dir.relative_to(REPO_ROOT)}"}
        try:
            ds_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", ds.source, str(ds_dir)],
                check=True, capture_output=True, text=True, timeout=600,
            )
        except FileNotFoundError:
            return {"status": "failed", "detail": "git is not available in this environment"}
        except subprocess.CalledProcessError as e:
            return {"status": "failed", "detail": f"git clone failed: {e.stderr.strip()[:300]}"}
        except subprocess.TimeoutExpired:
            return {"status": "failed", "detail": "git clone timed out"}
        return {"status": "success", "detail": f"cloned to {ds_dir.relative_to(REPO_ROOT)}"}

    try:
        files = resolve_files(ds)
    except Exception as e:
        return {"status": "failed", "detail": f"could not resolve file list: {e}"}

    if not files:
        return {"status": "failed", "detail": "no files found at source"}

    sizes_known = [s for _, _, s in files if s is not None]
    total_size = sum(sizes_known) if len(sizes_known) == len(files) else None
    if total_size is not None and max_bytes is not None and total_size > max_bytes:
        return {"status": "skipped",
                "detail": f"total size {total_size / 1e9:.2f} GB exceeds --max-size-gb {max_size_gb}"}

    ds_dir = out_dir / ds.id
    downloaded = 0
    try:
        for filename, url, _size in files:
            dest = ds_dir / filename
            if dest.exists():
                continue
            downloaded += download_stream(url, dest, max_bytes)
    except SizeExceeded:
        return {"status": "skipped",
                "detail": f"exceeded --max-size-gb {max_size_gb} mid-download; partial file removed"}
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        return {"status": "failed", "detail": f"network error: {e}"}
    except Exception as e:
        return {"status": "failed", "detail": str(e)}

    if downloaded == 0:
        return {"status": "success", "detail": f"already present at {ds_dir.relative_to(REPO_ROOT)}"}
    return {"status": "success",
            "detail": f"downloaded {downloaded / 1e9:.2f} GB to {ds_dir.relative_to(REPO_ROOT)}"}


# --------------------------------------------------------------------------
# State + markdown rendering
# --------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))


STATUS_ICON = {
    "success": "✅ Success",
    "failed": "❌ Failed",
    "skipped": "⏭️ Skipped",
    "manual": "\U0001f512 Manual",
    "pending": "⏳ Pending",
}


def render_markdown(state: dict) -> str:
    lines = [
        "# Breast Cancer Dataset Download Links & Status",
        "",
        "Auto-generated by [scripts/download_datasets.py](../scripts/download_datasets.py) — do not hand-edit",
        "the tables below; edit the `DATASETS` registry in the script instead and re-run it.",
        "",
        "Companion to [breast_cancer_datasets.md](breast_cancer_datasets.md) (which has full citations/",
        "modality detail per dataset); this file tracks *download attempts only*. Datasets already wired",
        "into this repo's training pipeline (mini-MIAS, BUS-BRA, BUSI, BUSC, BrEaST-Lesions USG, BreaDM)",
        "are intentionally excluded.",
        "",
        "**Status legend**: ✅ Success (fetched) · ❌ Failed (network/parse error, see Notes) · "
        "⏭️ Skipped (exceeded the size cap) · \U0001f512 Manual (needs login/DUA/specialized tool — "
        "see Notes for what to do) · ⏳ Pending (not attempted yet).",
        "",
        "MRI is deliberately downloaded last — its datasets are the largest in this inventory.",
        "",
    ]
    if state.get("_meta"):
        meta = state["_meta"]
        lines.append(f"_Last run: {meta.get('last_run', '?')} UTC · max-size-gb={meta.get('max_size_gb', '?')}_")
        lines.append("")

    categories = []
    for ds in DATASETS:
        if ds.category not in categories:
            categories.append(ds.category)

    for category in categories:
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Dataset | Link | Access | Status | Last Attempt (UTC) | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for ds in DATASETS:
            if ds.category != category:
                continue
            entry = state.get(ds.id, {})
            status = entry.get("status", "pending")
            status_display = STATUS_ICON.get(status, status)
            last_attempt = entry.get("last_attempt", "–")
            detail = entry.get("detail", ds.notes)
            lines.append(
                f"| {ds.name} | [link]({ds.info_url}) | {ds.access} | {status_display} | "
                f"{last_attempt} | {detail} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_markdown(state: dict) -> None:
    MD_PATH.write_text(render_markdown(state))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="List all datasets with their access classification and exit.")
    parser.add_argument("--init-md", action="store_true", help="Write docs/dataset_download_links.md with all datasets Pending, without downloading anything.")
    parser.add_argument("--all", action="store_true", help="Attempt every dataset, in registry order (MRI last).")
    parser.add_argument("--dataset", action="append", default=[], help="Attempt only this dataset id (repeatable).")
    parser.add_argument("--category", action="append", default=[], help="Attempt only datasets in this category (repeatable).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help=f"Output directory (default: {DEFAULT_OUT_DIR.relative_to(REPO_ROOT)}).")
    parser.add_argument("--max-size-gb", type=float, default=2.0, help="Per-dataset size cap in GB; larger datasets are skipped, not downloaded (default: 2.0). Use 0 for unlimited.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print what would be downloaded without writing any files.")
    args = parser.parse_args(argv)

    if args.list:
        for ds in DATASETS:
            print(f"{ds.id:28s} [{ds.access:12s}] {ds.category:16s} {ds.name}")
        return 0

    if args.init_md:
        write_markdown(load_state())
        print(f"Wrote {MD_PATH.relative_to(REPO_ROOT)} with {len(DATASETS)} datasets, all Pending.")
        return 0

    max_size_gb = None if args.max_size_gb == 0 else args.max_size_gb
    args.out = args.out.resolve()

    targets = DATASETS
    if args.dataset:
        wanted = set(args.dataset)
        targets = [d for d in DATASETS if d.id in wanted]
        missing = wanted - {d.id for d in targets}
        if missing:
            print(f"Unknown dataset id(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 1
    elif args.category:
        wanted = set(args.category)
        targets = [d for d in DATASETS if d.category in wanted]
    elif not args.all:
        parser.error("Pass --all, --dataset <id>, or --category <name> (or --list / --init-md).")

    state = load_state()

    for ds in targets:
        print(f"\n=== {ds.name} ({ds.id}) — access={ds.access} ===")
        if args.dry_run:
            if ds.access in ("zenodo", "figshare", "direct"):
                try:
                    files = resolve_files(ds)
                    total = sum(s or 0 for _, _, s in files)
                    print(f"  would fetch {len(files)} file(s), ~{total / 1e9:.2f} GB known")
                except Exception as e:
                    print(f"  could not resolve: {e}")
            else:
                print(f"  {ds.access}: {ds.notes}")
            continue

        result = attempt_download(ds, args.out, max_size_gb)
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        state[ds.id] = {"status": result["status"], "detail": result["detail"], "last_attempt": now}
        state["_meta"] = {"last_run": now, "max_size_gb": args.max_size_gb}
        save_state(state)
        write_markdown(state)
        print(f"  -> {result['status']}: {result['detail']}")
        time.sleep(0.5)  # be polite between requests

    if not args.dry_run:
        print(f"\nUpdated {MD_PATH.relative_to(REPO_ROOT)} after {len(targets)} dataset(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
