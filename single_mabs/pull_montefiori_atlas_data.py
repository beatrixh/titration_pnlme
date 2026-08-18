import pymssql
import pandas as pd, numpy as np
import labkey
from labkey.api_wrapper import APIWrapper


def get_study_list(lab="Montefiori"):
    if lab=="Montefiori":
        api = APIWrapper("atlas.scharp.org", "HVTN/Labs/Montefiori/Approved Runs", use_ssl=True)
    elif lab=="Mkhize":
        api = APIWrapper("atlas.scharp.org", "HVTN/Labs/Mkhize/NAb assays approved", use_ssl=True)
    else:
        print("Lab must be one of 'Montefiori' or 'Mkhize'")
    my_results = api.query.select_rows(
        schema_name="core",
        query_name="Containers",
        columns="CreatedBy,Created,Parent,Name,SortOrder,Description,Title,Searchable,Type,LockState,ExpirationDate,ID,FolderType,DisplayName,Path,ContainerType,Workbook,IdPrefixedName",
        container_filter="CurrentAndSubfolders"
    )
    
    df = pd.DataFrame(my_results['rows'])
    df = df[['Name', 'Path']]
    df['ParentDir'] = df.Path.str.rpartition("/", expand=True)[0].str.rpartition("/")[2]

    if lab=="Montefiori":
        studies = df.loc[df.ParentDir=="Approved Runs"].Name.tolist()
    elif lab=="Mkhize":
        studies = df.loc[df.ParentDir=="NAb assays approved"].Name.tolist()
        
    return studies


def get_study_data(study_list):
    usecols = [
    'ParticipantId',
    'ParticipantVisit/Visit',
    'SpecimenLsid/Property/VisitID',
    'SpecimenLsid/Property/Date',
    'Run/Name',
    'Run/VirusID',
    'Run/VirusName',
    'Run/VirusType',
    'Run/VirusDilution',
    'Run/IncubationTime',
    'DilutionData/Dilution',
    'DilutionData/PercentNeutralization',
    'DilutionData/MinDilution', #Min dilution 
    'DilutionData/MaxDilution', #Max dilution for the curve
    'DilutionData/Min', #Min signal across the replicates at a given dilution
    'DilutionData/Max', #Max signal across the replicates at a given dilution
    'SpecimenLsid/Property/initialdilution',
    'SpecimenLsid/Property/visitid',
    'SpecimenLsid/Property/specimenid',
    'SpecimenLsid/Property/concunits',
    'SpecimenLsid/Property/factor',
    'SpecimenLsid/Property/Method',
    'SpecimenLsid/Property/ConcUnits',
    ]
    
    data = {}
    
    for run in study_list:
        try:
            api = APIWrapper("atlas.scharp.org", f"HVTN/Labs/Montefiori/Approved Runs/{run}", use_ssl=True)
        
            my_results = api.query.select_rows(
                schema_name="study",
                query_name="DM_NAb_HVTN",
                columns=','.join(usecols)
            )
        
            data[run] = pd.DataFrame(my_results['rows'])
            data[run]['run'] = run
        except:
            print(f"Issue with {run}")
    return data