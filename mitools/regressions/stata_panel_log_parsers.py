import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import pandas as pd
from pandas import DataFrame

from mitools.regressions.quantiles import QuantilesRegressionSpecs

LogStructure = Type[str]  # type alias for log


def get_log_data(
    regression_info: QuantilesRegressionSpecs,
    log_path: Path,
    section_break: Optional[str] = None,
    subsection_break: Optional[str] = None,
):
    if section_break is None:
        section_break = "%$&" * 20
    if subsection_break is None:
        subsection_break = "#~^" * 20

    with open(log_path, "r") as f:
        log = f.read()

    log_structures = [s for s in log.split(section_break) if s]
    log_structures = [
        s.split(subsection_break) if subsection_break in s else s
        for s in log_structures
    ]

    log_data = {}
    descriptive_statistics = get_descriptive_statistics_from_log_structure(
        log_structures[0]
    )
    descriptive_statistics.index = descriptive_statistics.index.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    log_data["descriptive_statistics"] = descriptive_statistics

    skewness_tests = get_skewness_test_from_log_structure(log_structures[1])
    skewness_tests.index = skewness_tests.index.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    log_data["skewness_tests"] = skewness_tests

    correlations_table = get_correlations_table_from_log_structure(log_structures[2])
    correlations_table.index = correlations_table.index.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    correlations_table.columns = correlations_table.columns.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    log_data["correlations_table"] = correlations_table

    unit_tests = get_unit_tests_from_log_structure(log_structures[3])
    unit_tests["llc_specifications"].index = unit_tests["llc_specifications"].index.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    unit_tests["llc_results"].index = unit_tests["llc_results"].index.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    unit_tests["ips_specifications"].index = unit_tests["ips_specifications"].index.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    unit_tests["ips_results"].index = unit_tests["ips_results"].index.map(
        lambda x: (restore_variable(x[0], regression_info.variables), x[1])
    )
    log_data.update(unit_tests)

    cointegration_specifications, cointegration_results = (
        get_cointegration_test_from_log_structure(log_structures[4])
    )
    log_data["cointegration_specifications"] = cointegration_specifications
    log_data["cointegration_results"] = cointegration_results

    quantile_tables, regression_table = get_model_results_from_log_structure(
        log_structures[5]
    )
    quantile_tables.index = quantile_tables.index.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    regression_table = arrange_regression_table(regression_table, regression_info)
    log_data["quantile_tables"] = quantile_tables
    log_data["regression_table"] = regression_table

    vif_values = get_vif_from_log_structure(log_structures[6])
    vif_values.index = vif_values.index.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    log_data["vif_values"] = vif_values

    coeffs_correlations = get_coeffs_correlation_from_log_structure(log_structures[7])
    coeffs_correlations.index = coeffs_correlations.index.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    coeffs_correlations.columns = coeffs_correlations.columns.map(
        lambda x: restore_variable(x, regression_info.variables)
    )
    log_data["coeffs_correlations"] = coeffs_correlations

    return log_data


def arrange_regression_table(
    regression_table, regression_info: QuantilesRegressionSpecs
):
    regression_table = regression_table.unstack().to_frame()
    regression_table.index = regression_table.index.map(
        lambda x: (x[0], restore_variable(x[-1], regression_info.variables))
    )
    regression_table.columns = (
        ["Value"] if not regression_info.group else [regression_info.group]
    )
    regression_table["Variable Type"] = regression_table.index.get_level_values(
        "Independent Vars"
    ).map(
        lambda x: "Exog"
        if x in regression_info.independent_variables
        else "Control"
        if x in regression_info.control_variables
        else "Other"
    )
    regression_table = regression_table.reset_index()
    regression_table["Dependent Var"] = regression_info.dependent_variable
    regression_table["Regression Degree"] = (
        regression_info.quadratic if regression_info.quadratic else "linear"
    )
    regression_table["Regression Type"] = regression_info.regression_type
    regression_table["Id"] = regression_info.regression_id
    regression_table = regression_table.set_index(
        [
            "Id",
            "Regression Type",
            "Regression Degree",
            "Dependent Var",
            "Variable Type",
            "Independent Vars",
            "Quantile",
        ]
    )
    regression_table = regression_table.sort_index(
        ascending=[True, True, True, True, False, True, True]
    )
    return regression_table


def restore_variable(x, variables):
    if "~" not in x:
        return x
    x = x.split("~")
    return [var for var in variables if var.startswith(x[0]) and var.endswith(x[-1])][0]


def get_descriptive_statistics_from_log_structure(
    descriptive_statistics_structure: LogStructure,
) -> DataFrame:
    table_sections = [
        s.strip()
        for s in re.split(r"^-+$", descriptive_statistics_structure, flags=re.MULTILINE)
        if s.strip()
    ]
    descriptive_statistics = []
    for n, table in enumerate(table_sections):
        headers, values = re.split(r"-+\+-+", table)
        variables = [
            v
            for v in re.sub(r"\s+", " ", headers.replace("|", "")).split(" ")
            if v != "Stats" and v
        ]
        values = [
            [v for v in re.sub(r"\s+", " ", line.replace("|", "")).split(" ") if v]
            for line in values.split("\n")
            if line
        ]
        values = {vals[0]: vals[1:] for vals in values if set(vals[0]) != set("-")}
        table = pd.DataFrame(values, index=variables)
        descriptive_statistics.append(table)
    descriptive_statistics = pd.concat(descriptive_statistics, axis=0)
    return descriptive_statistics


def get_skewness_test_from_log_structure(log_structure) -> DataFrame:
    table_section = log_structure.split("----- Joint test -----")[1].strip()
    headers, values = re.split(r"-+\+-+", table_section)
    headers = headers.replace("Adj chi2(2)", "Adj-chi2(2)")
    variables = [
        v
        for v in re.sub(r"\s+", " ", headers.replace("|", "")).split(" ")
        if v != "Variable" and v
    ]
    values = [
        [v for v in re.sub(r"\s+", " ", line.replace("|", "")).split(" ") if v]
        for line in values.split("\n")
        if line
    ]
    values = {vals[0]: vals[1:] for vals in values if set(vals[0]) != set("-")}
    skewness_tests = pd.DataFrame(values, index=variables).T
    return skewness_tests


def get_correlations_table_from_log_structure(log_structure: LogStructure) -> DataFrame:
    table_sections = re.split(
        r"\n\s{13}\|\n\n\s{13}\|", log_structure.strip(), flags=re.MULTILINE
    )
    coefficients_tables = []
    pvalues_tables = []

    all_indexes = set()
    all_columns = set()

    for n, table in enumerate(table_sections):
        headers, table = re.split(r"-+\+-+", table)
        variables = [
            v
            for v in re.sub(r"\s+", " ", headers.replace("|", "")).split(" ")
            if v != "Variable" and v
        ]
        all_columns.update(variables)
        table = [l.strip() for l in table.split("\n") if l and set(l) != set(" |")]
        indexes = [row.split("|")[0].strip() for row in table if row[0].isalpha()]
        all_indexes.update(indexes)
        if n == 0:
            sorted_indexes = indexes
        coefficients_table = pd.DataFrame(index=indexes, columns=variables)
        pvalues_table = pd.DataFrame(index=indexes, columns=variables)
        for row in table:
            values = [v for v in row.split("|")[1].split(" ") if v]
            if row[0].isalpha():
                index = row.split("|")[0].strip()
                coefficients_table.loc[index, variables[: len(values)]] = values
            else:
                pvalues_table.loc[index, variables[: len(values)]] = values
        coefficients_tables.append(coefficients_table)
        pvalues_tables.append(pvalues_table)
    assert len(all_indexes) == len(all_columns), "Something went wrong"
    coefficients_tables = [
        table.reindex(index=all_indexes, columns=all_columns)
        for table in coefficients_tables
    ]
    pvalues_tables = [
        table.reindex(index=all_indexes, columns=all_columns)
        for table in pvalues_tables
    ]

    coefficients_table = coefficients_tables[0]
    for frame in coefficients_tables[1:]:
        coefficients_table = coefficients_table.combine_first(frame)
    pvalues_table = pvalues_tables[0]
    for frame in pvalues_tables[1:]:
        pvalues_table = pvalues_table.combine_first(frame)

    coefficients_table = coefficients_table.loc[sorted_indexes]
    coefficients_table = coefficients_table[
        coefficients_table.count().sort_values(ascending=False).index
    ]
    pvalues_table = pvalues_table.loc[sorted_indexes]
    pvalues_table = pvalues_table[
        pvalues_table.count().sort_values(ascending=False).index
    ]

    return coefficients_table


def get_unit_tests_from_log_structure(
    log_structure: LogStructure,
) -> Dict[str, DataFrame]:
    llc_unittests = log_structure[0::2]
    ips_unittests = log_structure[1::2]

    llc_specifications, llc_results = get_llc_unittest_data_from_log_structure(
        llc_unittests
    )
    ips_specifications, ips_results = get_ips_unittest_data_from_log_structure(
        ips_unittests
    )
    return dict(
        llc_specifications=llc_specifications,
        llc_results=llc_results,
        ips_specifications=ips_specifications,
        ips_results=ips_results,
    )


def get_llc_unittest_data_from_log_structure(
    log_structure: LogStructure,
) -> Tuple[DataFrame]:
    test_specifications = []
    test_results = []
    for section in log_structure:
        pattern = r"-+\n|-+$"
        matches = [match for match in re.finditer(pattern, section)]
        if len(matches) == 0:
            continue
        test = section[: matches[0].start()].strip()
        test_name = test.split("for")[0].strip()
        variable = test.split("for")[1].strip()

        test_specification = section[matches[0].end() : matches[1].start()].strip()
        test_specification = [line for line in test_specification.split("\n") if line]
        test_specification = [
            re.split(r"\s{2,}", line, 1)
            if len([c for c in line if c in [":", "="]]) == 2
            else line
            for line in test_specification
        ]
        test_specification = [
            item
            for sublist in test_specification
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        test_specification = [
            s.split(":") if ":" in s else s.split("=") for s in test_specification
        ]
        test_specification = {
            key.strip(): val.strip() for key, val in test_specification
        }
        test_specification = pd.DataFrame(test_specification, index=[variable])
        test_specification.index.name = test_name
        test_specifications.append(test_specification)

        test1_variables = section[matches[1].end() : matches[2].start()].strip()
        test1_variables = [
            line.strip() for line in re.split(r"\s+", test1_variables) if line
        ]

        test_result = section[matches[2].end() : matches[3].start()].strip()
        test_result = [
            [v.strip() for v in re.split(r"\s{2,}", vals) if v]
            for vals in test_result.split("\n")
            if vals
        ]
        test_result[0].append("") if len(test_result[0]) == 2 else test_result[0]
        test_result = {vals[0]: vals[1:] for vals in test_result}
        test_result = pd.DataFrame(test_result, index=test1_variables).T
        test_result.index = pd.MultiIndex.from_product([[variable], test_result.index])
        test_result.index.names = ["Variable", test_name]
        test_results.append(test_result)
    test_specifications = pd.concat(test_specifications, axis=0)
    test_results = pd.concat(test_results, axis=0)
    return test_specifications, test_results


def get_ips_unittest_data_from_log_structure(
    log_structure: LogStructure,
) -> Tuple[DataFrame]:
    test_results = []
    test_specifications = []
    for n, section in enumerate(log_structure):
        pattern = r"-+\n|-+$"
        matches = [match for match in re.finditer(pattern, section)]
        test = section[: matches[0].start()].strip()
        test_name = test.split("for")[0].strip()
        variable = test.split("for")[1].strip()
        test_specification = section[matches[0].end() : matches[1].start()].strip()
        test_specification = [line for line in test_specification.split("\n") if line]
        test_specification = [
            re.split(r"\s{2,}", line, 1)
            if len([c for c in line if c in [":", "="]]) == 2
            else line
            for line in test_specification
        ]
        test_specification = [
            item
            for sublist in test_specification
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        test_specification = [
            s.split(":") if ":" in s else s.split("=") for s in test_specification
        ]
        test_specification = {
            key.strip(): val.strip() for key, val in test_specification
        }
        if test_specification["Panel means"].find("sequentially"):
            test_specification["Panel means"] = test_specification["Panel means"].split(
                " "
            )[0]
            test_specification["Asymptotics"] = (
                test_specification["Asymptotics"] + " sequentially"
            )
        test_specification = pd.DataFrame(test_specification, index=[variable])
        test_specification.index.name = test_name
        test_specifications.append(test_specification)
        test_variables = section[matches[1].end() : matches[2].start()].strip()
        test_variables = [line for line in test_variables.split("\n") if line]
        header, test_variables = test_variables[0], test_variables[1]
        test_variables = [
            line.strip() for line in re.split(r"\s{2,}", test_variables) if line
        ]
        test_variables = [
            v if v in ["Statistic", "p-value"] else f"{header} {v}"
            for v in test_variables
        ]
        test_result = section[matches[2].end() :].strip()
        _test_result = [line.strip() for line in test_result.split("\n") if line]
        test_result = {}
        for n, line in enumerate(_test_result):
            values = re.split(r"\s{2,}", line)
            if n == 0:
                if not "(Not available)" in line:
                    _section = section
                    test_result[values[0]] = [values[1], ""] + values[2:]
                else:
                    test_result[values[0]] = [values[1], "", "", "", ""]
            elif n == 1:
                test_result[values[0]] = [values[1], "", "", "", ""]
            elif n == 2:
                test_result[values[0]] = values[1:3] + ["", "", ""]
        test_result = pd.DataFrame(test_result, index=test_variables).T
        test_result.index.name = test_name
        test_result.index = pd.MultiIndex.from_product([[variable], test_result.index])
        test_result.index.names = ["Variable", test_name]
        test_results.append(test_result)

    test_specifications = pd.concat(test_specifications, axis=0)
    test_results = pd.concat(test_results, axis=0)
    return test_specifications, test_results


def get_cointegration_test_from_log_structure(
    log_structure: LogStructure,
) -> DataFrame:
    cointegration_test = log_structure.strip()

    test_name = cointegration_test.split(" ")[0]
    n_panels = "".join(
        [
            c
            for c in cointegration_test.split("Number of panels")[1].split("\n")[0]
            if c.isdigit()
        ]
    )
    if "Number of periods" in cointegration_test:
        periods_split = "Number of periods"
    elif "Avg. number of periods" in cointegration_test:
        periods_split = "Avg. number of periods"
    n_periods = "".join(
        [
            c
            for c in cointegration_test.split(periods_split)[1].split("\n")[0]
            if c.isdigit() or c == "."
        ]
    )

    test_specifications = re.split(
        r"\n-+", cointegration_test.split(f"{n_periods}\n")[1]
    )[0]
    pattern = re.compile(
        r"([A-Za-z]+(?: [A-Za-z]+)?):\s*([^:]+?)(?=(?:\s+[A-Za-z]+(?: [A-Za-z]+)?\:|$))"
    )
    results = pattern.findall(test_specifications)
    test_specifications = {key.strip(): value.strip() for key, value in results}
    test_specifications = pd.DataFrame(test_specifications, index=[test_name])
    test_specifications["Number of panels"] = n_panels
    test_specifications["Number of periods"] = n_periods
    test_specifications = test_specifications.T

    pattern = r"-+\n"
    matches = [match for match in re.finditer(pattern, cointegration_test)]
    test_variables = cointegration_test[matches[1].end() : matches[2].start()].strip()
    test_variables = [
        v for v in re.sub(r"\s+", " ", test_variables.replace("|", "")).split(" ") if v
    ]

    test_result = re.split(r"\n-+", cointegration_test[matches[2].end() :])[0].strip()
    test_result = [
        [v.strip() for v in re.split(r"\s{2,}", vals) if v]
        for vals in test_result.split("\n")
        if vals
    ]
    test_result = {
        vals[0]: vals[1:] for vals in test_result if set(vals[0]) != set("-")
    }
    test_result = pd.DataFrame(test_result, index=test_variables).T

    return test_specifications, test_result


def get_model_results_from_log_structure(
    log_structure: LogStructure,
) -> Tuple[DataFrame]:
    table_sections = re.split(
        r"\.\d+ Quantile regression$", log_structure.strip(), flags=re.MULTILINE
    )
    n_obs, table_sections = table_sections[0], table_sections[1:]
    quantiles = re.findall(
        r"\.\d+ Quantile regression$", log_structure.strip(), flags=re.MULTILINE
    )

    quantile_tables = []
    for quantile, table in zip(quantiles, table_sections):
        table = re.sub(r"^-+$", "", table, flags=re.MULTILINE)
        headers, values = re.split(r"-+\+-+", table)
        headers = headers.replace("Std. err.", "Std.err.")
        headers = headers.replace("[95% conf. interval]", "[95%conf.interval]")
        variables = [
            v for v in re.sub(r"\s+", " ", headers.replace("|", "")).split(" ") if v
        ]
        variables[-1] = variables[-1] + "_left"
        variables.append(variables[-1].replace("left", "right"))
        values = [
            [v for v in re.sub(r"\s+", " ", line.replace("|", "")).split(" ") if v]
            for line in values.split("\n")
            if line
        ]
        values = {vals[0]: vals[1:] for vals in values if set(vals[0]) != set("-")}
        regression_table = pd.DataFrame(values, index=variables).T
        regression_table.columns = pd.MultiIndex.from_product(
            [[float(quantile.split(" ")[0])], regression_table.columns],
            names=["Quantile", "value"],
        )
        regression_table.index.name = "Independent Vars"
        quantile_tables.append(regression_table)
    quantile_tables = pd.concat(quantile_tables, axis=1)

    def significance_map(value):
        if value <= 0.001:
            return "***"
        if value <= 0.01:
            return "**"
        if value <= 0.05:
            return "*"
        return ""

    regression_table = quantile_tables.copy().astype(float)
    significances = regression_table.loc[:, (slice(None), "P>z")].applymap(
        lambda x: significance_map(x)
    )
    significances.columns = pd.MultiIndex.from_tuples(
        [(col[0], "significance") for col in significances.columns],
        names=["Quantile", "value"],
    )
    regression_table = pd.concat([regression_table, significances], axis=1).sort_index(
        axis=1
    )
    regression_table.loc[:, (slice(None), "z")] = (
        regression_table.loc[:, (slice(None), "z")]
        .astype(str)
        .applymap(lambda x: f"({x})")
    )
    regression_table.loc[:, (slice(None), "Coefficient")] = regression_table.loc[
        :, (slice(None), "Coefficient")
    ].astype(str)
    regression_table.loc[:, (slice(None), "Coefficient")] += regression_table.loc[
        :, (slice(None), "z")
    ].values
    regression_table.loc[:, (slice(None), "Coefficient")] += regression_table.loc[
        :, (slice(None), "significance")
    ].values
    regression_table = regression_table.loc[:, (slice(None), "Coefficient")].droplevel(
        1, axis=1
    )

    return quantile_tables, regression_table


def get_vif_from_log_structure(log_structure: LogStructure) -> DataFrame:
    vif_variables = re.split(r"-+\+-+\n", log_structure)[0].strip()
    vif_variables = [v for v in vif_variables.split(" ") if v and v != "|"]

    pattern = r"-+\+-+\n"
    matches = [match for match in re.finditer(pattern, log_structure)]
    vif_values = log_structure[matches[0].end() : matches[1].start()]
    vif_values = [
        [v for v in line.split(" ") if v and v != "|"]
        for line in vif_values.split("\n")
        if line
    ]
    vif_values = {v[0]: v[1:] for v in vif_values}
    vif_values = pd.DataFrame(vif_values, index=vif_variables[1:]).T
    vif_values.index.name = vif_variables[0]

    return vif_values


def get_coeffs_correlation_from_log_structure(log_structure: LogStructure) -> DataFrame:
    table_sections = re.split(
        r"(?<=[\d\w])\s*\n\n\s{8}", log_structure.strip(), flags=re.MULTILINE
    )[1:]
    coefficients_tables = []

    all_columns = set()
    all_indexes = set()

    for n, section in enumerate(table_sections):
        headers, table = re.split(r"-+\+-+", section)
        variables = [
            v
            for v in re.sub(r"\s+", " ", headers.replace("|", "")).split(" ")
            if v != "e(V)" and v
        ]
        all_columns.update(variables)
        table = [l.strip() for l in table.split("\n") if l and set(l) != set(" |")]
        indexes = [row.split("|")[0].strip() for row in table if row[0].isalpha()]
        all_indexes.update(indexes)
        if n == 0:
            sorted_indexes = indexes
        coefficients = pd.DataFrame(index=indexes, columns=variables)
        for row in table:
            values = [v for v in row.split("|")[1].split(" ") if v]
            if row[0].isalpha():
                index = row.split("|")[0].strip()
            coefficients.loc[index, variables[: len(values)]] = values
        coefficients_tables.append(coefficients)
    assert len(all_indexes) == len(all_columns), "Something went wrong"
    coefficients_tables = [
        table.reindex(index=all_indexes, columns=all_columns)
        for table in coefficients_tables
    ]
    coefficients_table = coefficients_tables[0]
    for frame in coefficients_tables[1:]:
        coefficients_table = coefficients_table.combine_first(frame)
    coefficients_table = coefficients_table.loc[sorted_indexes]
    coefficients_table = coefficients_table[
        coefficients_table.count().sort_values(ascending=False).index
    ]

    return coefficients_table
