import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import pandas as pd
from pandas import DataFrame

LogStructure = Type[str]  # type alias for log


def get_log_data(
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

    descriptive_statistics = get_descriptive_statistics_from_log_structure(
        log_structures[0]
    )
    skewness_tests = get_skewness_test_from_log_structure(log_structures[1])
    correlations_table = get_correlations_table_from_log_structure(log_structures[2])
    unit_tests = get_unit_tests_from_log_structure(log_structures[3])
    cointegration_specifications, cointegration_results = (
        get_cointegration_test_from_log_structure(log_structures[4])
    )
    quantile_tables, regression_table = get_model_results_from_log_structure(
        log_structures[5]
    )
    vif_values = get_vif_from_log_structure(log_structures[6])
    coeffs_correlations = get_coeffs_correlation_from_log_structure(log_structures[7])

    return (
        descriptive_statistics,
        skewness_tests,
        correlations_table,
        unit_tests,
        cointegration_specifications,
        cointegration_results,
        quantile_tables,
        regression_table,
        vif_values,
        coeffs_correlations,
    )


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
    test1_results = []
    test1_specifications = []
    test2_results = []
    test2_specifications = []

    for section in log_structure:
        if section and section != "\n":
            pattern = r"-+\n|-+$"
            matches = [match for match in re.finditer(pattern, section)]

            test1 = section[: matches[0].start()].strip()

            test1_name = test1.split("for")[0].strip()
            variable = test1.split("for")[1].strip()

            test1_specification = section[matches[0].end() : matches[1].start()].strip()
            test1_specification = [
                line for line in test1_specification.split("\n") if line
            ]
            test1_specification = [
                re.split(r"\s{2,}", line, 1)
                if len([c for c in line if c in [":", "="]]) == 2
                else line
                for line in test1_specification
            ]
            test1_specification = [
                item
                for sublist in test1_specification
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]
            test1_specification = [
                s.split(":") if ":" in s else s.split("=") for s in test1_specification
            ]
            test1_specification = {
                key.strip(): val.strip() for key, val in test1_specification
            }
            test1_specification = pd.DataFrame(test1_specification, index=[variable])
            test1_specification.index.name = test1_name
            test1_specifications.append(test1_specification)

            test1_variables = section[matches[1].end() : matches[2].start()].strip()
            test1_variables = [
                line.strip() for line in re.split(r"\s+", test1_variables) if line
            ]

            test1_result = section[matches[2].end() : matches[3].start()].strip()
            test1_result = [
                [v.strip() for v in re.split(r"\s{2,}", vals) if v]
                for vals in test1_result.split("\n")
                if vals
            ]
            test1_result[0].append("") if len(test1_result[0]) == 2 else test1_result[0]
            test1_result = {vals[0]: vals[1:] for vals in test1_result}
            test1_result = pd.DataFrame(test1_result, index=test1_variables).T
            test1_result.index = pd.MultiIndex.from_product(
                [[variable], test1_result.index]
            )
            test1_result.index.names = ["Variable", test1_name]
            test1_results.append(test1_result)

            test2 = section[matches[3].end() : matches[4].start()].strip()
            test2_name = test2.split("for")[0].strip()
            variable = test2.split("for")[1].strip()

            test2_specification = section[matches[4].end() : matches[5].start()].strip()
            test2_specification = [
                line for line in test2_specification.split("\n") if line
            ]
            test2_specification = [
                re.split(r"\s{2,}", line, 1)
                if len([c for c in line if c in [":", "="]]) == 2
                else line
                for line in test2_specification
            ]
            test2_specification = [
                item
                for sublist in test2_specification
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]
            test2_specification = [
                s.split(":") if ":" in s else s.split("=") for s in test2_specification
            ]
            test2_specification = {
                key.strip(): val.strip() for key, val in test2_specification
            }
            if test2_specification["Panel means"].find("sequentially"):
                test2_specification["Panel means"] = test2_specification[
                    "Panel means"
                ].split(" ")[0]
                test2_specification["Asymptotics"] = (
                    test2_specification["Asymptotics"] + " sequentially"
                )
            test2_specification = pd.DataFrame(test2_specification, index=[variable])
            test2_specification.index.name = test2_name
            test2_specifications.append(test2_specification)

            test2_variables = section[
                matches[5].end() + 1 : matches[6].start() + 1
            ].strip()
            test2_variables = [line for line in test2_variables.split("\n") if line]
            header, test2_variables = test2_variables[0], test2_variables[1]
            test2_variables = [
                line.strip() for line in re.split(r"\s{2,}", test2_variables) if line
            ]
            test2_variables = [
                v if v in ["Statistic", "p-value"] else f"{header} {v}"
                for v in test2_variables
            ]

            test2_result = section[matches[6].end() + 1 :].strip()
            _test2_result = [line.strip() for line in test2_result.split("\n") if line]
            test2_result = {}
            for n, line in enumerate(_test2_result):
                values = re.split(r"\s{2,}", line)
                if n == 0:
                    test2_result[values[0]] = [values[1], ""] + values[2:]
                elif n == 1:
                    test2_result[values[0]] = [values[1], "", "", "", ""]
                elif n == 2:
                    test2_result[values[0]] = values[1:3] + ["", "", ""]
            test2_result = pd.DataFrame(test2_result, index=test2_variables).T
            test2_result.index.name = test2_name
            test2_result.index = pd.MultiIndex.from_product(
                [[variable], test2_result.index]
            )
            test2_result.index.names = ["Variable", test2_name]
            test2_results.append(test2_result)

    test1_specifications = pd.concat(test1_specifications, axis=0)
    test1_results = pd.concat(test1_results, axis=0)
    test2_specifications = pd.concat(test2_specifications, axis=0)
    test2_results = pd.concat(test2_results, axis=0)

    return dict(
        test1_specifications=test1_specifications,
        test1_results=test1_results,
        test2_specifications=test2_specifications,
        test2_results=test2_results,
    )


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
    n_periods = "".join(
        [
            c
            for c in cointegration_test.split("Number of periods")[1].split("\n")[0]
            if c.isdigit()
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
            names=["quantile", "value"],
        )
        regression_table.index.name = "Variable"
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
        names=["quantile", "value"],
    )
    regression_table = pd.concat([regression_table, significances], axis=1).sort_index(
        axis=1
    )
    regression_table.loc[:, (slice(None), "z")] = (
        regression_table.loc[:, (slice(None), "z")]
        .astype(str)
        .applymap(lambda x: f" ({x})")
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
