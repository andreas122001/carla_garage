def simple_table_line(d):
    mean = d["Total"]["scores_mean"]
    std = d["Total"]["scores_std"]

    latex = "Agent "
    for key in ["score_route", "score_penalty", "score_composed"]:
        latex += f" & {mean[key]:.1f}{{\\scriptsize\\textcolor{{gray}}{{$\pm${std[key]:.1f}}}}}"
    latex += " \\\\\\hline"
    print(latex)


def overall_results(d):
    score_order = ["score_route", "score_penalty", "score_composed"]
    # Remove the average
    average = d["Total"]
    del d["Total"]

    scenarios = sorted(d.keys())

    # For each scenario
    latex = """\
\\begin{table}[htp!]
    \centering
    \\begin{tabular}{l|c|c|c}
        \\Xhline{1pt}
        Scenario & RC $\\uparrow$ & IS $\\uparrow$ & DS $\\uparrow$\\\\\\hline\n"""

    for scenario in scenarios:
        latex += f"        {scenario}"

        mean = d[scenario]["scores_mean"]
        std = d[scenario]["scores_std"]
        for key in score_order:
            latex += f" & {mean[key]:.2f}{{\\scriptsize\\textcolor{{gray}}{{$\pm${std[key]:.2f}}}}}"

        latex += "\\\\\n"
    latex += "    \\hline\n"
    latex += "    \\textbf{Average}"
    for key in score_order:
        latex += f" & {average['scores_mean'][key]:.2f}{{\\scriptsize\\textcolor{{gray}}{{$\pm${average['scores_std'][key]:.2f}}}}}"
    latex += """\
\\\\
    \\Xhline{1pt}
    \\end{tabular}
    \\caption{Caption}
    \\label{tab:key}
\\end{table}
"""

    print()
    print()
    print(latex)
    print()
    print()


def overall_results_comparison(d1, d2):
    score_order = ["score_route", "score_penalty", "score_composed"]
    d = [d1, d2]
    # Remove the average
    average = [d1["Total"], d2["Total"]]

    scenarios = sorted(d1.keys())
    scenarios.remove("Total")

    # For each scenario
    latex = """\
\\begin{table}[h]
    \centering
    \caption{Example Table with Combined Rows in Headers}
    \setlength{\\arrayrulewidth}{0.0001pt}
    \\renewcommand{\\arraystretch}{1.2}
    \\footnotesize{
    \\begin{tabularx}{\\textwidth}{lp{0pt} XXX p{0pt} XXX}
        \\Xhline{1pt}
        \\multirow{2}{*}{Scenario} && \\multicolumn{3}{c}{\\textbf{TF++ Expert}} && \multicolumn{3}{c}{\\textbf{PDM-Lite}} \\\\
        \\cline{3-5}\\cline{7-9}
        && RC $\\uparrow$ & IS $\\uparrow$ & DS $\\uparrow$ && RC $\\uparrow$ & IS $\\uparrow$ & DS $\\uparrow$ \\\\
        \\hline
"""

    for scenario in scenarios:
        latex += f"        {scenario}"

        for i in range(2):
            latex += f"\n        &"
            mean = d[i][scenario]["scores_mean"]
            std = d[i][scenario]["scores_std"]
            for key in score_order:
                latex += f"& {mean[key]:.1f}{{\\tiny\\textcolor{{gray}}{{$\pm${std[key]:.1f}}}}} "

        latex += "\\\\\n\n"
    latex += "        \\hline\n"
    latex += "        \\textbf{Average}"
    for i in range(2):
        latex += f"\n        &"
        for key in score_order:
            latex += f"& {average[i]['scores_mean'][key]:.1f}{{\\tiny\\textcolor{{gray}}{{$\pm${average[i]['scores_std'][key]:.1f}}}}} "
    latex += "\\\\\n"
    latex += """\
        \\Xhline{1pt}
    \\end{tabularx}
    }
\\end{table}
"""

    print()
    print()
    print(latex)
    print()
    print()


import ujson

# with open("eval_default_town13.json") as f:
#     d1 = ujson.load(f)
# with open("eval_pdml_town13.json") as f:
#     d2 = ujson.load(f)

# with open("tfpp_default_model_results.json") as f:
#     d1 = ujson.load(f)
# with open("tfpp-pdm_lite-max_speed21_lb2_results.json") as f:
#     d2 = ujson.load(f)

# overall_results_comparison(d1, d2)
simple_table_line(
    {
        "Total": {
        "scores_mean": {
            "score_route": 99.92665789473683,
            "score_penalty": 0.9287680268421051,
            "score_composed": 92.83328601052634
        },
        "scores_std": {
            "score_route": 0.44612260978792573,
            "score_penalty": 0.06691674974273573,
            "score_composed": 6.71040464448619
        }
    }
    }
)
