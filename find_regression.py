import subprocess
from timeit import default_timer as timer
import json
import os

data_file = "/localdisk/amyskov/benchmark_datasets/regression/yellow_tripdata_2015-01.csv"
# data_file = "/localdisk/amyskov/benchmark_datasets/regression/test.csv"
s3_path = "s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-01.csv"
iterations_number = 3
good_commit = "4d35e7076a938b2cc327ce8d6e1922dd69bcff00"
report_file = "regression_report.txt"
report_metric_file = "regression_metric_report.txt"
report_modin_good = "modin_good_results.txt"
max_g2b_deviation = 0.8 # good/bad

def execute_cmd(cmd: list):
    print(f"CMD: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f"stdout: {process.communicate()[0]}\n stderr: {process.communicate()[1]}")
    return process.communicate()

def get_git_revision_hash(encoding="ascii"):
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], encoding=encoding).strip()

def measure(func, *args, **kw):
    t0 = timer()
    res = func(*args, **kw)
    t = timer() - t0
    return res, t

def q_read_csv(pd):
    res = pd.read_csv(data_file, parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"], quoting=3)
    res.shape
    return res

def q_reductions(df):
    res = df.count()
    repr(res)
    return res

def q_map_operations(df):
    res = df.isnull()
    repr(res)
    return res

def q_appy(df):
    res = df["trip_distance"].apply(round)
    repr(res)
    return res

def q_add_column(df, col):
    df["rounded_trip_distance"] = col
    repr(df)
    return df

def q_groupby_agg(df):
    res = df.groupby(by="rounded_trip_distance").count()
    repr(res)
    return res

def bench(pd):
    results = {}
    df, results["t_read_csv"] = measure(q_read_csv, pd)
    _, results["t_reductions"] =  measure(q_reductions, df)
    _, results["t_map_operations"] = measure(q_map_operations, df)
    col_apply, results["t_apply"] = measure(q_appy, df)
    _, results["t_add_column"] = measure(q_add_column, df, col_apply)
    _, results["t_groupby_agg"] = measure(q_groupby_agg, df)

    return results

def bench_iterations(iterations=1):
    results = {}
    final_results = {}
    
    for i in range(1, iterations+1):
        import modin.pandas as pd
        pd.DEFAULT_NPARTITIONS = 4
        results[f"iteration_{i}"] = bench(pd)

    for t in results["iteration_1"].keys():
        values = []
        for it in results.keys():
            values.append(results[it][t])

        final_results[t] = min(values)

    return final_results

def make_report(result, report_file=report_file):
    with open(report_file, mode = 'a') as file:
        file.write(json.dumps(result) + "\n")

def metric(good_results: dict, bad_results: dict):
    coefficients = {}
    for t in good_results.keys():
        assert t in bad_results.keys()
        coefficients[t] = good_results[t] / bad_results[t]

    count = 0
    _sum = 0
    for coeff_key in coefficients:
        count += 1
        _sum += coefficients[coeff_key]

    mean = _sum / count
    good = mean > max_g2b_deviation
    coefficients["version"] = get_git_revision_hash()
    coefficients["mean"] = mean

    return coefficients, good

def eval_modin_bench(good_results):
    reults = bench_iterations(iterations=iterations_number)
    metrics, good = metric(good_results, reults)

    reults["version"] = get_git_revision_hash()

    make_report(reults, report_file=report_file)
    make_report(metrics, report_file=report_metric_file)
    print("reults: \n", reults)
    print("metrics: \n", metrics)

    return good

def remove_old_reports():
    for f in [report_file, report_metric_file]:
        if os.path.exists(f):
            os.remove(f)

def get_good_results(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
    
    return data
    

if __name__ == "__main__":
    # remove_old_reports()

    # reults_modin_good = bench_iterations(iterations=iterations_number)
    # make_report(reults_modin_good, report_file=report_modin_good)

    good_results = get_good_results(report_modin_good)
    good = eval_modin_bench(good_results)
    print("good", good)
