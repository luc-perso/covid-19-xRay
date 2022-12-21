import os
import pandas as pd

# data names
lung_name = 'lung'
infection_name = 'infection'

train_name = 'train'
test_name = 'test'
valid_name = 'valid'

no_covid_name = 'no_covid'
covid_name = 'covid19'
normal_name = 'normal'

images_name = 'images'
masks_name = 'masks'
lung_mask_name = lung_name + '_' + masks_name
infection_mask_name = infection_name + '_' + masks_name

# data directory names
lung_dirname = 'Lung Segmentation Data'
infection_dirname = 'Infection Segmentation Data'

train_dirname = 'Train'
test_dirname = 'Test'
valid_dirname = 'Val'

no_covid_dirname = 'Non-COVID'
covid_dirname = 'COVID-19'
normal_dirname = 'Normal'

images_dirname = 'images'
lung_masks_dirname = 'lung masks'
infection_masks_dirname = 'infection masks'

db_names = pd.DataFrame(data={
    'db_name': [lung_name, infection_name],
    'db_dirname': [lung_dirname, infection_dirname]
})
ds_names = pd.DataFrame(data={
    'ds_name': [train_name, test_name, valid_name],
    'ds_dirname': [train_dirname, test_dirname, valid_dirname]
})
desease_names = pd.DataFrame(data={
    'desease_name': [no_covid_name, covid_name, normal_name],
    'desease_dirname': [no_covid_dirname, covid_dirname, normal_dirname]
})
data_type_names = pd.DataFrame(data={
    'data_type_name': [images_name, lung_mask_name, infection_mask_name],
    'data_type_dirname': [images_dirname, lung_masks_dirname, infection_masks_dirname]
})

# Dataset


def build_data_paths():
    db_names['keys'] = 1
    ds_names['keys'] = 1
    desease_names['keys'] = 1
    data_type_names['keys'] = 1

    # db_names
    data_paths = pd.merge(db_names, ds_names, on='keys')
    data_paths = pd.merge(data_paths, desease_names, on='keys')
    data_paths = pd.merge(data_paths, data_type_names, on='keys')
    data_paths = data_paths.drop(columns=['keys'])
    # data_paths
    data_paths = data_paths.set_index(
        ['db_name', 'ds_name', 'desease_name', 'data_type_name'])
    data_paths['path'] = data_paths.apply(
        lambda x: os.path.join(*(x.to_list())), axis=1)
    data_paths = data_paths.iloc[:, -1:]

    # data_paths
    idx = pd.IndexSlice
    index = data_paths.loc[idx[[lung_name], :, :, infection_mask_name]].index
    data_paths = data_paths.drop(index)

    return data_paths


