import datetime
from itertools import product
from pull_montefiori_atlas_data import *

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os


def main():
    # pull list of study names
    mntf_studies = get_study_list()

    # manual download of atlas data
    atlas_data=pd.read_csv("/networks/vtn/lab/SDMC_labscience/operations/projects/nab_curves/plate_data/atlas_data_from_api.csv", index_col=[0,1])
    # subset to concentration only (get rid of dilution data)
    atlas_data = atlas_data.loc[atlas_data.method=='Concentration']
    atlas_runs = atlas_data.run.unique().tolist()

    # merge plate locs and save each study
    formatted_dir = "/networks/vtn/lab/SDMC_labscience/operations/projects/nab_curves/plate_data/protocols_formatted_2026-06-10/"
    for run in atlas_runs:
        print(run)
        sdata = atlas_data.loc[atlas_data.run==run]
        platedir = f'/networks/vtn/lab/SDMC_labscience/operations/projects/nab_curves/plate_data/protocols/{run_map[str(run)]}/assaydata/'
        dfs = []
        for f in [i for i in os.listdir(platedir) if i[-4:]=="xlsx"]:
            try:
                dfs+= [merge_plate_loc(platedir+f, sdata)]
            except:
                pass
        if len(dfs)>0:
            tmp = pd.concat(dfs)
            tmp = tmp.drop_duplicates()
            if len(tmp)>0:
                tmp.to_csv(formatted_dir + f"plate_data_{run}.csv", index=False)

    # concat all studies
    formatted_dir = '/networks/vtn/lab/SDMC_labscience/operations/projects/nab_curves/plate_data/protocols_formatted_2026-06-10/'
    dataset = pd.concat([pd.read_csv(formatted_dir+f) for f in os.listdir(formatted_dir)])

    # merge metadata back on
    additional_metadata = atlas_data[['run_id','run','virus_name','virus_id']].drop_duplicates()
    dataset = dataset.merge(additional_metadata, on=['run_id'], how='left')

    # virus and cell controls should have zero mAb
    dataset['concentration'] = dataset.id_unique.str.rpartition("|")[2]
    dataset.loc[dataset.specrole!='sample', 'concentration'] = 0.

    # monolix wants a unique id per concentration. for controls, need to add in plate_row
    dataset['monolix_id'] = dataset.specrole + "|" + dataset.plate_row.astype(str)
    dataset.loc[dataset.specrole=='sample','monolix_id'] = dataset.specrole
    dataset['monolix_id'] = dataset[['run','run_id','monolix_id','plate_col','virus_name','virus_id']].astype(str).agg("|".join, axis=1)

    dataset.concentration = dataset.concentration.astype(float)

    # normalize RLUs per plate. will re-correct after monolix fits
    dataset['rlu_max'] = dataset.groupby('run_id').rlu.transform("max")
    dataset['rlu_min'] = dataset.groupby('run_id').rlu.transform("min")
    dataset['rlu_norm'] = (dataset.rlu - dataset.rlu_min) / (dataset.rlu_max - dataset.rlu_min)

    dataset['virus_col'] = dataset.virus_name
    dataset.loc[dataset.virus_col.isna(), 'virus_col'] = dataset.loc[dataset.virus_col.isna(), 'virus_id']

    # binary virus present marker (for monolix)
    dataset['virus_cc'] = dataset.specrole.map({'sample':1, 'cc':0, 'vc':1})

    # merge on metadata
    mab_metadata = dataset.id_unique.str.split("|", expand=True)[[4,5]].rename(columns={4:'mab',5:'mablot'})
    dataset = pd.concat([dataset, mab_metadata], axis=1)
    dataset['mab_virus'] = dataset[['mab','virus_col']].astype(str).agg("|".join, axis=1)

    # filter down to rows that are unique per monolix_id/concentration/row/col
    only_once = dataset.groupby(['monolix_id','run_id','specrole','plate_col','plate_row']).rlu_norm.count()
    only_once = only_once.loc[only_once==1].reset_index().monolix_id.unique().tolist()
    dataset.loc[dataset.monolix_id.isin(only_once)].shape, dataset.shape
    dataset_pre_filter = dataset.copy()
    dataset = dataset.loc[dataset.monolix_id.isin(only_once)]

    ## select the ones that have any neutralization
    square = dataset.pivot(
        index=['monolix_id','run_id','specrole','plate_col'],
        columns='plate_row',
        values='rlu_norm'
    )

    square['variance'] = square[np.arange(8)+1].var(axis=1)
    square['goes_down'] = square[np.arange(3)+1].mean(axis=1)*0.85 > square[[6,7,8]].mean(axis=1)

    square = square.reset_index()
    square.loc[square.specrole!='sample','goes_down'] = False
    goes_down_ids = square.loc[square.goes_down].monolix_id.tolist()

    # mark data that has any neutralization vs none (flat line + edge effects)
    dataset['goes_down'] = False
    dataset.loc[dataset.monolix_id.isin(goes_down_ids), 'goes_down'] = True


    # save data
    savedir = 'input_data/'
    dataset.to_csv(
        savedir+'atlas_data_with_plate_locs_2026-06-12.csv', index=False
    )

    # pull out run_ids for which an entire plate was run
    full_plates = dataset.groupby('run_id').count()[['run_name']]
    full_plates = full_plates.loc[full_plates.run_name==96].index.tolist()
    full_plates = dataset.loc[dataset.run_id.isin(full_plates)]
    non_full_plates = dataset.loc[~dataset.run_id.isin(full_plates.run_id)]

    full_plates.to_csv(
        savedir+'atlas_data_with_plate_locs_subset_full_plates_2026-06-14.csv', index=False
    )

    # pull out single mAbs
    combos = [i for i in full_plates.mab_virus.unique() if "+" in i]
    single_mabs_version = dataset.loc[~dataset.mab_virus.isin(combos)]
    single_mabs_version.to_csv(
        savedir+'atlas_full_data_single_mabs_with_plate_locs_2026-07-28.csv', index=False
    )




    # something different got saved to this name
    # full_plates_single_mabs_version = full_plates.loc[~full_plates.mab_virus.isin(combos)]
    # full_plates_single_mabs_version.to_csv(
    #     savedir+'atlas_data_single_mabs_with_plate_locs_2026-07-30.csv', index=False
    # )

    #larger than the full plates dataset, but not the full full dataset.
    #a large chunk of the mostly full plates + the half full
    some_neutralization = dataset.loc[(
        (dataset.specrole.isin(['cc','vc'])) | (dataset.goes_down==True) & ~(dataset.mab_virus.isin(combos))
    )]

    min56 = some_neutralization.groupby('run_id').count().rlu_norm
    min56 = min56.loc[min56 >= 56].reset_index().run_id.unique().tolist()

    min56 = some_neutralization.loc[(
        some_neutralization.run_id.isin(min56)
    )]

    more_full = some_neutralization.groupby('run_id').count().rlu_norm
    more_full = more_full.loc[(more_full == 48 )].reset_index().run_id.unique().tolist()

    extra_run_ids = np.random.choice(more_full, size=(20_000 - len(min56))//48 + 1, replace=False)

    subset = pd.concat([
        some_neutralization.loc[some_neutralization.run_id.isin(extra_run_ids)],
        min56
    ])

    subset.to_csv(
        savedir + "atlas_data_single_mabs_plate_locs_20k_rows_2026_08_07.csv", index=False
    )

    ## saving subsets to try to diagnose issue
    min96 = some_neutralization.groupby('run_id').count().rlu_norm
    min96 = min96.loc[min96 >= 96].reset_index().run_id.unique().tolist()

    subset_minus_full_plates = subset.loc[~subset.run_id.isin(min96)].run_id.unique().tolist()[:150]
    subset_minus_full_plates = subset.loc[subset.run_id.isin(subset_minus_full_plates)]
    subset_minus_full_plates.to_csv(
        savedir + "subset_minus_96well_7200_rows_2026_08_08.csv", index=False
    )

    subset_minus_full_plates = subset.loc[~subset.run_id.isin(min96)].run_id.unique().tolist()[150:]
    subset_minus_full_plates = subset.loc[subset.run_id.isin(subset_minus_full_plates)]
    subset_minus_full_plates.to_csv(
        savedir + "subset_minus_96well_10215_rows_2026_08_08.csv", index=False
    )

    very_small_subset = np.random.choice(more_full, size=10, replace=False)
    very_small_subset = pd.concat([
        some_neutralization.loc[some_neutralization.run_id.isin(very_small_subset)],
        min56
    ])
    very_small_subset.to_csv(
        savedir + "very_small_subset_2026_08_08.csv", index=False
    )

    subset_10k_rows = np.random.choice(more_full, size=10_000 // 48, replace=False)
    subset_10k_rows = some_neutralization.loc[some_neutralization.run_id.isin(subset_10k_rows)]
    subset_10k_rows.to_csv(
        savedir + "subset_10k_rows_2026_08_08.csv", index=False
    )

    # revised subset
    vc_rlu_map = original.loc[original.specrole=='vc'].groupby('run_id').rlu.mean()
    dataset['mean_vc'] = dataset.run_id.map(vc_rlu_map)

    ids = [
        1740832,
        1752396,
        1706657,
        1706687,
        1706717,
        1706723,
        1742458,
        1838189,
        1838381,
        1843721,
        1200432,
        1313842,
        1342384,
        1313860,
        1342396,
        1313884,
        1342408,
        1313920,
        1342240,
        1313938,
        1342258,
        1313956,
        1342276,
        1313974,
        1342294,
        1313992,
        1342312,
        1314016,
        1342336,
        1314040,
        1342348,
        1314058,
        1342372,
    ]
    revised_subset = dataset.loc[(
        (dataset.run_id.isin(ids))
    )]

    revised_subset.to_csv(
        savedir + "revised_subset_full_plates_some_neutralization_2026-08-10.csv", index=False
    )


column_rename = {
    'Run':'run_id',
    'Run/Name':'run_name',
    'Run/VirusName':'virus_name',
    'SpecimenLsid/Property/MAbLot':'mablot',
    'ParticipantId':'mab',
    'DilutionData/MinDilution':'min_dilution', #Min dilution 
    'DilutionData/Dilution':'concentration',
    'DilutionData/PercentNeutralization':'pct_neutralization',
    'Run/VirusID':'virus_id',
    'SpecimenLsid/Property/specimenid':'guspec',
    'SpecimenLsid/Property/ConcUnits':'concentration_units',
    'SpecimenLsid/Property/Method':'method',
    'Run/FileID':'run_fileid',
    'Run/VirusControlAggregates/AvgValue': 'virus_control_mean',
    'Run/VirusControlAggregates/StdDevValue': 'virus_control_std',
    'DilutionData/Min':'sample_min',
    'DilutionData/Max':'sample_max',
    'DilutionData/Mean': 'sample_rlu_mean',
    'DilutionData/StdDev': 'sample_rlu_std',
    'Run/CellControlAggregates/AvgValue': 'cell_control_mean',
    'Run/CellControlAggregates/StdDevValue': 'cell_control_std',
    'FitParameters': 'Fit Parameters',
    'FitError': 'Fit Error',
    'DilutionData/ReplicateName':'replicate_name',
    'Run/PlateNumber':'plate_number',
    'Run/Experiment_Date':'experiment_date',
}

run_map = {
    '704_VRC01_Concordance': '704_VRC01_Concordance',
    '206_mAb': '206',
    '115 mAb': '115_mAb',
    '301_mAb': '301',
    '130_Clinical': '130_Clinical',
    '302_mAb': '302',
    '704': '704',
    '130_bNAb_Combinations': '130_bNAb_Combinations',
    '703 Concordance': '703',
    '127_clinical': '127',
    '305_mAb': '305',
    '704 Pilot_cross protocol_clinical': '704_Pilot_cross_protocol',
    '140_mAb': '140',
    '136_Clinical': '136',
    '804': '804',
    '704_pilot': '704_pilot',
    '104': '104',
    '100 FH1 Antibody': '100_FH1_Antibody',
    '704_breakthrough_mAbs': '704_breakthrough_mAbs'
}

def merge_plate_loc(plate_path, data):
    cols = range(3,13)
    def pick_random_col(row):
        candidates = row.index[row.values]
        return np.random.choice(candidates) 
    plate = pd.read_excel(plate_path, sheet_name=None)
    if 'Plate' in plate.keys():
        plate = plate['Plate']
        plate = plate.iloc[:,[2,5]]
        plate.columns = ['well','rlu']
        plate['row_val'] = plate.well.str[0]
        plate['col_val'] = plate.well.str[1:].astype(int)
        
        rowmap = {i:j for (i,j) in zip(plate.row_val.unique(),range(1,9))}
        plate.row_val = plate.row_val.map(rowmap)
        
        # convert to plate-formatted
        plate = plate.pivot(
            index='row_val',
            columns='col_val',
            values='rlu'
        )
    elif 'Results' in plate.keys():
        plate = plate['Results']
    
        a_locs = np.where(plate.astype(str).eq('A'))
        plate = plate.iloc[a_locs[0][0]:, a_locs[1][0]+1:]
        
        plate.columns=np.arange(12)+1
        plate['row_val'] = np.arange(8)+1
        plate=plate.set_index('row_val')
    else:
        print(plate_path)
        return pd.DataFrame()
    
    sdata = data[['run_id','run_name','id_unique','row_val','col_val','sample_min','sample_max']].melt(
        id_vars=['run_id','run_name','id_unique','row_val','col_val'],
        value_vars=['sample_min','sample_max']
    )

    #merge plate onto atlas data
    sdata = sdata.merge(
        plate,
        on='row_val',
        how='left'
    )

    diffs = sdata[cols].sub(sdata['value'], axis=0).abs()
    # get row-wise minimum
    min_vals = diffs.min(axis=1)
    
    # find all columns that match the minimum
    is_min = diffs.eq(min_vals, axis=0)
    
    sdata['closest_col'] = is_min.apply(pick_random_col, axis=1) #todo: just choose first one instead of random
    sdata['min_diff'] = min_vals
    
    min_diffs = sdata.groupby('run_id').min_diff.sum()
    least, idx = min_diffs.min(), min_diffs.argmin()
    # there should be one run for which we get all matches, else no match
    if least > 1:
        # print(f"no match found :( {plate_path}")
        return pd.DataFrame()

    # if we found a run that matches the plate data, subset atlas data to that run
    run_id = min_diffs.index[idx]
    sdata = sdata.loc[sdata.run_id==run_id]

    # now we'll check for matches again, but this time we'll see if we got multiple hits
    diffs = sdata[cols].sub(sdata['value'], axis=0).abs()
    min_vals = diffs.min(axis=1)
    is_min = diffs.eq(min_vals, axis=0)
    
    sdata['candidate_cols'] = [
        list(diffs.columns[row])
        for row in is_min.to_numpy()
    ]
    
    candidate_lists = [
        list(diffs.columns[row])
        for row in is_min.to_numpy()
    ]
    
    for assignment in product(*candidate_lists):
        test = sdata.copy()
        test['closest_col'] = assignment

        # with this assignment, per run/row are is the count of unique column assignments the same as the count of records
        valid = (
            test.groupby(['run_id', 'row_val'])['closest_col']
                .nunique()
                .eq(test.groupby(['run_id', 'row_val']).size())
                .all()
        )
    
        if valid:
            # print("match found")
            sdata = test
            break
    else:
        # print("No valid assignment found")
        return pd.DataFrame()
        # subset down to desired cols
    sdata = sdata[['run_id','run_name','id_unique','row_val','closest_col','value','min_diff']]
    sdata = sdata.rename(columns={'row_val':'plate_row', 'closest_col':'plate_col', 'value':'rlu'})
    sdata['specrole'] = 'sample'

    run_id, run_name = sdata[['run_id','run_name']].iloc[0]
    
    # merge on controls
    cc = pd.DataFrame({
        'run_id':[run_id]*8,
        'run_name':[run_name]*8,
        'specrole':['cc']*8,
        'plate_col':[1]*8,
        'plate_row':plate.index.tolist(),
        'rlu':plate[1]
    })
    
    vc = pd.DataFrame({
        'run_id':[run_id]*8,
        'run_name':[run_name]*8,
        'specrole':['vc']*8,
        'plate_col':[2]*8,
        'plate_row':plate.index.tolist(),
        'rlu':plate[2]
    })
    df = pd.concat([sdata, cc, vc])
    df['plate_path'] = plate_path
    return df

if __name__=="__main__":
    main()