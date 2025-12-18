# trainer_test.py
import threading

import pandas as pd
import sklearn

from kepler_model.train import load_class
from kepler_model.util.loader import default_train_output_pipeline
from kepler_model.util.train_types import PowerSourceMap, default_trainer_names
from tests.extractor_test import (
    get_expected_power_columns,
    get_extract_results,
    node_info_column,
    test_energy_source,
    test_extractors,
)
from tests.isolator_test import get_isolate_results, test_isolators

test_trainer_names = default_trainer_names
pipeline_lock = threading.Lock()


def assert_train(trainer, data, energy_components):
    trainer.print_log("assert train")
    node_types = pd.unique(data[node_info_column])
    for node_type in node_types:
        node_type_str = int(node_type)
        node_type_filtered_data = data[data[node_info_column] == node_type]
        X_values = node_type_filtered_data[trainer.features].values
        for component in energy_components:
            try:
                output = trainer.predict(node_type_str, component, X_values)
                assert len(output) == len(X_values), f"length of predicted values != features ({len(output)}!={len(X_values)})"
            except sklearn.exceptions.NotFittedError:
                pass


def process(node_level, feature_group, result, trainer_names=test_trainer_names, energy_source=test_energy_source, power_columns=get_expected_power_columns(), pipeline_name=default_train_output_pipeline):
    energy_components = PowerSourceMap[energy_source]
    train_items = []
    for trainer_name in trainer_names:
        trainer_class = load_class("trainer", trainer_name)
        trainer = trainer_class(energy_components, feature_group, energy_source, node_level=node_level, pipeline_name=pipeline_name)
        trainer.process(result, power_columns, pipeline_lock=pipeline_lock)
        assert_train(trainer, result, energy_components)
        train_items += [trainer.get_metadata()]
    return pd.concat(train_items)


def process_all(extractors=test_extractors, isolators=test_isolators, trainer_names=test_trainer_names, energy_source=test_energy_source, power_columns=get_expected_power_columns(), pipeline_name=default_train_output_pipeline):
    abs_train_list = []
    dyn_train_list = []
    for extractor in extractors:
        extractor_name = extractor.__class__.__name__
        extractor_results = get_extract_results(extractor_name, node_level=True)
        for feature_group, result in extractor_results.items():
            print("Extractor ", extractor_name)
            metadata_df = process(True, feature_group, result, trainer_names=trainer_names, energy_source=energy_source, power_columns=power_columns, pipeline_name=pipeline_name)
    return metadata_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer Test")
    parser.add_argument("--dataset_path", type=str, help="Path of dataset", default="ready_4_train")
    parser.add_argument("--energy_source", type=str, help="Energy source", default='meter')
    parser.add_argument("--cross_validation", action='store_true', help="Cross validation", default=False)
    parser.add_argument("--feature_group", type=str, help="Feature group: All7, _6_tx, _5_tx_irq, _4_tx_irq_page", default="All7")

    args = parser.parse_args()

    save_path = os.path.join(data_path, args.dataset_path)
    abs_train_df = process_all(energy_source=args.energy_source, feature_group=args.feature_group, save_path=save_path, cross_validation=args.cross_validation)
    print('trainer results:', abs_train_df[["model_name","mae","mape"]].sort_values(by=["mae","mape"], ascending=[True,True]))