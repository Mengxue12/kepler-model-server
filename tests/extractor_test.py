# extractor_test.py
# - extractor.extract
#
# To use response:
# from extractor_test import get_extract_results
# extract_results = get_extract_results(extractor_name, feature_group, node_level)

# import external src
import os

from kepler_model.train import DefaultExtractor, SmoothExtractor
from kepler_model.train.pipeline import load_class
from kepler_model.util import (
    FeatureGroup,
    FeatureGroups,
    PowerSourceMap,
    assure_path,
    get_valid_feature_group_from_queries,
    load_csv,
    save_csv,
)
from kepler_model.util.extract_types import component_to_col
from kepler_model.util.prom_types import node_info_column
from kepler_model.util.train_types import all_feature_groups
from tests.prom_test import get_query_results, prom_output_path, prom_output_filename
import argparse

data_path = os.path.join(os.path.dirname(__file__), "data")
assure_path(data_path)
extractor_output_path = os.path.join(data_path, "extractor_output")

if not os.path.exists(extractor_output_path):
    os.mkdir(extractor_output_path)

test_extractors = [DefaultExtractor(), SmoothExtractor()]

test_energy_source = "rapl-sysfs"
test_energy_components = PowerSourceMap[test_energy_source]
test_num_of_unit = 1
test_customize_extractors = []


def get_filename(extractor_name, feature_group, node_name, node_level, meter_data_date=None):
    if meter_data_date is not None:
        return f"{extractor_name}_{feature_group}_{node_name}_{node_level}_{meter_data_date}"
    return f"{extractor_name}_{feature_group}_{node_name}_{node_level}"


def get_extract_result(extractor_name, feature_group, node_name, node_level, save_path=extractor_output_path, meter_data_date=None):
    filename = get_filename(extractor_name, feature_group, node_name, node_level, meter_data_date=meter_data_date)
    file_path = os.path.join(save_path, filename + ".csv")
    print("get_extract_result: Loading extract result from", file_path)
    if not os.path.exists(file_path):
        print(f"get_extract_result: File {filename} does not exist. Returning None.")
        return None
    return load_csv(save_path, filename)


def get_extract_results(extractor_name, node_name, node_level, save_path=extractor_output_path, meter_data_date=None):
    all_results = dict()
    for feature_group in all_feature_groups:
        print("get_extract_results: Getting extract result for feature group:", feature_group)
        result = get_extract_result(extractor_name, feature_group, node_name, node_level, save_path=save_path, meter_data_date=meter_data_date)
        if result is not None:
            all_results[feature_group] = result
    return all_results


def save_extract_results(instance, feature_group, extracted_data, node_name, node_level, save_path=extractor_output_path, meter_data_date=None):
    extractor_name = instance.__class__.__name__
    filename = get_filename(extractor_name, feature_group, node_name, node_level, meter_data_date=meter_data_date)
    print("saving extracted csv to ", save_path, filename)
    save_csv(save_path, filename, extracted_data)


def get_expected_power_columns(energy_components=test_energy_components, num_of_unit=test_num_of_unit):
    # TODO: if ratio applied,
    # return [component_to_col(component, "package", unit_val) for component in energy_components for unit_val in range(0,num_of_unit)]
    return [component_to_col(component) for component in energy_components]


def assert_extract(extracted_data, power_columns, energy_components, num_of_unit, workload_features, meter_data_path=None):
    extracted_data_column_names = extracted_data.columns
    # basic assert
    assert extracted_data is not None, "extracted data is None"
    assert len(power_columns) > 0, f"no power label column {extracted_data_column_names}"
    assert node_info_column in extracted_data_column_names, f"no {node_info_column} in column {extracted_data_column_names}"
    # TODO: if ratio applied, expected_power_column_length = len(energy_components) * num_of_unit
    expected_power_column_length = len(energy_components)
    # detail assert
    assert len(power_columns) == expected_power_column_length, f"unexpected power label columns {power_columns}, expected {expected_power_column_length}"
    # TODO: if ratio applied, expected_col_size must + 1 for power_ratio
    if meter_data_path is not None:
        num_of_unit += 1
    expected_col_size = expected_power_column_length + len(workload_features) + num_of_unit  # power ratio
    assert len(extracted_data_column_names) == expected_col_size, f"unexpected column length: expected {expected_col_size}, got {extracted_data_column_names}({len(extracted_data_column_names)}) "


def process(query_results, feature_group, node_name,
            save_path=extractor_output_path, 
            customize_extractors=test_customize_extractors, 
            energy_source=test_energy_source, 
            num_of_unit=test_num_of_unit, 
            meter_data_path=None):
    energy_components = PowerSourceMap[energy_source]
    print("process extract arguments: \n feature_group:", feature_group, "\n save_path:", save_path, "\n customize_extractors:", customize_extractors, "\n energy_source:", energy_source, "\n num_of_unit:", num_of_unit, "\n meter_data_path:", meter_data_path)
    global test_extractors
    for extractor_name in customize_extractors:
        test_extractors += [load_class("extractor", extractor_name)]
    for test_instance in test_extractors:
        print("Extractor:", test_instance.get_name())
        meter_data_date = None
        if meter_data_path is not None:
            meter_data_date = meter_data_path.split("/")[-1].split(".")[0]
        # node level
        extracted_data, power_columns, corr, workload_features, feature_data = test_instance.extract(query_results, energy_components, feature_group, energy_source, node_level=True, meter_data_path=meter_data_path)
        assert_extract(extracted_data, power_columns, energy_components, num_of_unit, workload_features, meter_data_path=meter_data_path)
        save_extract_results(test_instance, feature_group, extracted_data, node_name, True, save_path=save_path, meter_data_date=meter_data_date)
        print("is node_level:", True, "Correlations:\n", corr)
        # container level
        node_level = False
        if not node_level:
            meter_data_date = None
            meter_data_path = None
        extracted_data, power_columns, corr, workload_features, feature_data = test_instance.extract(query_results, energy_components, feature_group, energy_source, node_level=node_level, meter_data_path=meter_data_path)
        assert_extract(extracted_data, power_columns, energy_components, num_of_unit, workload_features, meter_data_path=meter_data_path)
        save_extract_results(test_instance, feature_group, extracted_data, node_name, node_level, save_path=save_path, meter_data_date=meter_data_date)
        print("is node_level:", False)
        print("Correlations:\n")
        print(corr)
        break


def test_extractor_process(
        node_name, 
        prom_output_path=prom_output_path, 
        prom_output_filename=prom_output_filename, 
        save_path=extractor_output_path, 
        energy_source=test_energy_source,
        meter_data_path=None):
    query_results = get_query_results(save_path=prom_output_path, save_name=prom_output_filename)
    # filter query results by node_name
    query_results = {
        key: df[df["instance"] == node_name] if "instance" in df.columns else df
        for key, df in query_results.items()
    }
    print("get query results of instance:", node_name)
    print("get energy source:", energy_source)
    assert len(query_results) > 0, "cannot read_sample_query_results"
    valid_feature_groups = get_valid_feature_group_from_queries(query_results.keys())
    print("valid feature groups:", valid_feature_groups)
    for fg in valid_feature_groups:
        feature_group = fg.name
        if feature_group not in ['WorkloadOnly', 'CounterOnly']:
            continue  # TODO only test workload and cpu features for now
        print('='*10, "feature_group:", feature_group, '='*10)
        process(
            query_results, feature_group, node_name,
            save_path=save_path,  
            energy_source=energy_source,
            meter_data_path=meter_data_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Run extractor tests.")
    parser.add_argument("--node_name", type=str, required=True, help="Name of the node to extract results.")
    parser.add_argument("--prom_output_path", type=str, default=prom_output_path, help="Path to the Prometheus output")
    parser.add_argument("--prom_output_filename", type=str, default=prom_output_filename, help="Filename of the Prometheus output")
    parser.add_argument("--save_path", type=str, default=extractor_output_path, help="Path to save the extracted results. Default is 'extractor_output' directory.")
    parser.add_argument("--energy_source", type=str, default=test_energy_source, help="Energy source to use for extraction. Default is 'rapl-sysfs'.")
    parser.add_argument("--meter_data_path", type=str, default=None, help="Path to the meter data CSV file. If provided, it will be used for extracting power data.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_extractor_process(
        args.node_name, 
        prom_output_path=args.prom_output_path,
        prom_output_filename=args.prom_output_filename,
        save_path=os.path.join(data_path, args.save_path), 
        energy_source=args.energy_source,
        meter_data_path=args.meter_data_path)