
import os
import numpy as np
import pandas as pd
from kepler_model.util import FeatureGroup, FeatureGroups, assure_path
from kepler_model.train.extractor.extract_meter_power import extract_meter_power
from prom_test import get_query_results
import argparse
import psutil
from extractor_test import data_path

pd.set_option("display.max_columns", None)
# pd.set_option("display.max_colwidth", None)
# pd.set_option("display.max_rows", None)
proc = psutil.Process(os.getpid())

def parse_args():
    parser = argparse.ArgumentParser(description="Run extractor tests.")
    parser.add_argument("--node_name", type=str, required=True, help="Name of the node to extract results.")
    parser.add_argument("prom_output_path", type=str, help="Path to the Prometheus output")
    parser.add_argument("prom_output_filename", type=str, help="Filename of the Prometheus output")
    parser.add_argument("--save_folder", type=str, help="Path to save the extracted results. Default is 'extractor_output' directory.")
    parser.add_argument("energy_source", type=str, help="Energy source to use for extraction. Default is 'rapl-sysfs'.")
    parser.add_argument("--meter_data_path", type=str, default=None, help="Path to the meter data CSV file. If provided, it will be used for extracting power data.")
    parser.add_argument("extract_level", type=str, help="node, container or process level.")
    return parser.parse_args()

def avg_values(df):
    diffs = df.index.to_series().diff()
    diffs.iloc[0] = 3
    mask = diffs != 3
    
    # 判断mask是否为空（没有True值）
    if not mask.any():  # 如果mask中没有任何True值
        print("All timestamps are recorded at 3s intervals - no missing data")
        df = df.div(diffs, axis=0)
        print("corr:\n")
        print(df.corr()[["platform_joules"]])
        return df
    
    # 如果有缺失的时间戳
    result = pd.concat([df[mask], diffs[mask]], axis=1)
    result['num_missing_rows'] = result['timestamp']/3-1
    print("the number of the values that is not recorded at 3s:", len(result))
    print("Missing timestamp details:")
    print(result)
    print("the number of the values that is missing:", result['num_missing_rows'].sum())
    
    df = df.div(diffs, axis=0)
    print("corr:\n")
    print(df.corr()[["platform_joules"]])
    return df

def remove_0_rows(df):
    mask = df.eq(0).all(axis=1)
    
    # 判断mask是否为空（没有全为0的行）
    if not mask.any():  # 如果mask中没有任何True值
        print("No rows with all zeros found")
        return df
    
    print(f"Found {len(df[mask])} rows with all zeros:")
    print(df[mask])
    
    df["group"] = (df["platform_joules"] != 0).cumsum()   # 不为0的个数
    # 每组：取第一个的值，最后一个的 index
    def take_first_with_last_index(g):
        # first = g.head(1).copy()               # 第一个内容
        # first.index = [g.tail(1).index[0]]     # 把 index 改成最后一个的
        avg = g.mean(numeric_only=True).to_frame().T
        # index 改成该组最后一行的 index
        avg.index = [g.tail(1).index[0]]
        return avg
    
    # 保存原始index的名字
    original_index_name = df.index.name
    
    result = (
        df.groupby("group").apply(take_first_with_last_index).reset_index(level=0, drop=True) 
    )
    
    # 恢复原始index的名字
    result.index.name = original_index_name

    result = result.drop(columns="group")      # 清理临时列
    return result

if __name__ == "__main__":
    args = parse_args()
    node_name = args.node_name
    energy_source = args.energy_source
    extract_level = args.extract_level
    meter_data_path = args.meter_data_path
    save_folder=args.save_folder
    
    query_results = get_query_results(args.prom_output_path, args.prom_output_filename)    # process data as dataFrames
    # filter query results by node_name
    print("Memory (MB):", proc.memory_info().rss / 1024 / 1024)
    query_results = {
        key: df[df["instance"] == node_name] if "instance" in df.columns else df
        for key, df in query_results.items()
    }
    print("Memory (MB):", proc.memory_info().rss / 1024 / 1024)
    print("get query results of instance:", node_name)
    print("get energy source:", energy_source)
    assert len(query_results) > 0, "cannot read_sample_query_results"
    feature_group = 'WorkloadOnly'
    fg = FeatureGroup[feature_group]
    features = FeatureGroups[fg]
    print("features", features)

    #####################################################################
    # 优化：使用列表推导式和更高效的数据处理
    processed_data = []
    features.append("platform_joules")
    
    # 预计算查询字符串，避免重复字符串操作
    # platform_joules_query = f'kepler_{extract_level}_platform_joules_total'
    

    if extract_level == "node":
        for feature in features:
            query = f'kepler_{extract_level}_{feature}_total'
            if query not in query_results:
                print(query, "not in", list(query_results.keys()))
                continue
                
            df = query_results[query]
            if len(df) == 0:
                print("no data in ", query)
                continue
                
            # 优化：使用更高效的零值检查
            if df[query].sum() == 0:
                print("all values in query", query, "are 0")
                continue
            query_data = df
            if feature == "platform_joules":
                # query_data=query_data.rename(columns={query:f'kepler_{extract_level}_platform_millijoules_total'})
                # query = f'kepler_{extract_level}_platform_millijoules_total'
                # 优化：使用in-place操作减少内存分配
                # query_data = query_data.copy()  # 只在需要修改时才复制
                query_data[query] = query_data[query] / 1000
            # 优化：直接选择需要的列，避免重复操作
            query_data.rename(columns={query: feature}, inplace=True)
            # for node level
            query_data = query_data[['timestamp', feature]].groupby('timestamp').sum()
            query_data = query_data.diff() # missing values and the time difference are dealt at line 226
            print(f"obtained {query} data")
            processed_data.append(query_data)
        processed_data = pd.concat(processed_data, axis=1)
        processed_data = processed_data.iloc[1:]  # 移除第一行NA行
        # 优化：使用更高效的字符串操作
        # prefix_to_remove = f"kepler_{extract_level}_"
        # suffix_to_remove = "_total"
        # df.columns = [col.removeprefix(prefix_to_remove).removesuffix(suffix_to_remove) for col in df.columns]
            
        processed_data = avg_values(processed_data) # divided
        processed_data = remove_0_rows(processed_data)
    elif extract_level == "container":
        container_df_map = dict()
        missing_timestamp = []
        print_missing_timestamp = True
        for feature in features:
            query = f'kepler_{extract_level}_{feature}_total'
            if query not in query_results:
                print(query, "not in", list(query_results.keys()))
                continue
                
            df = query_results[query]
            if len(df) == 0:
                print("no data in ", query)
                continue
                
            if df[query].sum() == 0:
                print("all values in query", query, "are 0")
                continue
                
            query_data = df
            # if feature != "platform_joules" and feature != "cpu_cycles": # TEST
            #     continue
            
            if feature == "platform_joules":
                query_data[query] = query_data[query] / 1000
                
            query_data.rename(columns={query: feature}, inplace=True)

            # for container
        
            # print("query_data.columns", query_data.columns)
            # print(f"query_data shape {feature}", query_data.shape)
            container_id_cols = ["container_id", "container_name", "pod_name", "container_namespace"]                
            query_data["id"] = query_data[container_id_cols].apply(lambda x: "/".join([str(xi) for xi in x]), axis=1)
            query_data = query_data[['timestamp', 'id', feature]]
            container_id_list = pd.unique(query_data["id"])
            print("number of containers:", len(container_id_list), "for feature", feature)

            container_query_data = []
            for container_id in container_id_list:
                container_df = query_data[query_data["id"] == container_id]
                # sum over mode
                container_df = container_df.groupby(['timestamp']).sum()
                # print(f"container_df of {container_id.split('/')[1:]} shape after grouping by timestamp", container_df.shape)
                time_diff_values = container_df.reset_index()[['timestamp']].diff().values
                time_diff_values[0] = 3
                mask = time_diff_values != 3
                if len(container_df[mask]) > 0:
                    timestamps = container_df[mask].reset_index()['timestamp']
                    new_missing = timestamps[~timestamps.isin(missing_timestamp)]
                    if len(new_missing) > 0:
                        missing_timestamp.extend(new_missing)
                        print(f"the number of the values that is not recorded at 3s for container {container_id}:", len(container_df[mask]))                
                        print(f"container_df[mask]\n", container_df[mask])
                # divided by time difference
                container_df[[feature]] = container_df[[feature]].diff()
                container_df[[feature]] = container_df[[feature]] / time_diff_values
                container_df = container_df.dropna()
                # container_query_data +=[container_df]
                if container_id in container_df_map:
                    # previously found container
                    # print(f"previously found container {container_id}")
                    container_df_map[container_id] = pd.concat([container_df_map[container_id], container_df[[feature]]], axis=1)
                else:
                    # newly found container
                    container_df_map[container_id] = container_df
                    # print("container_df_map keys", container_df_map.keys())
        for container_id, container_df in container_df_map.items():
            container_df['id'] = container_id
            processed_data += [container_df]
        processed_data = pd.concat(processed_data)
        # print("processed_data\n", processed_data)
        
        # query_data.to_csv(os.path.join(data_path, save_folder, f"{node_name}_processed_{extract_level}_{feature}_query_data.csv"))
    else:
        print("not support")
        exit()
        # print(f"shape of the query {query}: {query_data.shape}")
    ########################################################
    print("Memory (MB):", proc.memory_info().rss / 1024 / 1024)
    del query_results
    print("Memory (MB):", proc.memory_info().rss / 1024 / 1024)
    #####################################################

    if not processed_data.empty:
        save_path = os.path.join(data_path, save_folder) 
        assure_path(save_path)
        processed_data.to_csv(os.path.join(save_path, f"{node_name}_processed_{extract_level}_features.csv"))
        
    else:
        print("No processed data available")
    #####################################################
    if meter_data_path is not None:
        if extract_level != 'node':
            assert "meter data can only join node level data"
            exit()
        df = processed_data
        del processed_data
        print("Join the feature data with power data from meter by timestamp...")
        
        meter_power_data = extract_meter_power(meter_data_path)
        
        if not df.empty:
            start_idx, end_idx = df.index[0], df.index[-1]
            meter_power_data = meter_power_data[(meter_power_data.index >= start_idx) & 
                                              (meter_power_data.index <= end_idx)]
        
        if len(meter_power_data) > 0 and not df.empty:
            # 使用高效的向量化方法计算每个df时间戳区间内meter数据的平均值
            # 基于pandas的cut和groupby操作
            
            df_timestamps = df.index.values
            meter_timestamps = meter_power_data.index.values
            
            # 使用向量化的方法：基于searchsorted进行区间分配
            # 为每个df时间戳找到对应的meter数据区间，并计算平均值
            
            # 使用searchsorted找到每个df时间戳在meter数据中的位置
            right_positions = np.searchsorted(meter_timestamps, df_timestamps, side='right')
            
            # 计算每个区间的平均值
            aligned_values = []
            for i, right_pos in enumerate(right_positions):
                if i == 0:
                    # 第一个区间：从开始到第一个df时间戳
                    start_idx = 0
                    end_idx = right_pos
                else:
                    # 后续区间：从前一个df时间戳到当前df时间戳
                    start_idx = right_positions[i-1]
                    end_idx = right_pos
                
                if start_idx < end_idx and end_idx <= len(meter_power_data):
                    # 计算区间平均值
                    interval_data = meter_power_data.iloc[start_idx:end_idx]
                    avg_value = interval_data.mean()
                else:
                    # 如果没有数据，使用前一个值或第一个值
                    if aligned_values:
                        avg_value = aligned_values[-1]
                    else:
                        avg_value = meter_power_data.iloc[0] if len(meter_power_data) > 0 else pd.Series()
                
                aligned_values.append(avg_value)
            
            # 创建对齐后的DataFrame
            meter_power_data = pd.DataFrame(aligned_values, index=df.index)
            
            print("the shape of meter power after averaging", meter_power_data.shape)
            
            # 由于已经基于df.index对齐，不需要再进行索引交集操作
            print("Meter power data aligned with feature data based on timestamps")
            
            # 优化：直接concat，避免使用list
            df = pd.concat([df, meter_power_data], axis=1)
            df = df.iloc[1:] # TODO way to not remove
            
            # 优化：只对需要重命名的列进行操作
            prefix_to_remove = f"kepler_{extract_level}_"
            suffix_to_remove = "_total"
            df.columns = [col.removeprefix(prefix_to_remove).removesuffix(suffix_to_remove) 
                         for col in df.columns]

            print("corr:\n")
            print(df.corr()[["platform_joules", "meter-platform_power"]])
            
            # print("/3=================\n", (df/3).iloc[:, -3:])
            df.to_csv(os.path.join(save_path, f"{node_name}_processed_{extract_level}.csv"))
        else:
            print("No meter power data available")
        
