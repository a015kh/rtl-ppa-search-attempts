import json
import pandas as pd

from real_evolutionary_report_generator import fitness_score

if __name__ == "__main__":
    with open("data/synth/test_ppa_feedback.json", "r") as f:
        ppa_feedback = json.load(f)
    
    clock = 1.0
    names = [
        "Prob018_mux256to1__Gen15__Prob018_mux256to1_M-S_sample2",
        "Prob021_mux256to1v__Gen9__Prob021_mux256to1v_M-R_sample7",
        "Prob023_vector100r__Gen19__Prob023_vector100r_M-R_sample8",
        "Prob030_popcount255__Gen1__Prob030_popcount255_M-S_sample7",
        "Prob052_gates100__Gen6__Prob052_gates100_C-F_sample9",
        "Prob055_conditional__Gen1__Prob055_conditional_M-E_sample5",
        "Prob068_countbcd__Gen3__Prob068_countbcd_C-F_sample10",
        "Prob080_timer__Gen0__Prob080_timer_initial_sample6",
        "Prob092_gatesv100__Gen8__Prob092_gatesv100_M-S_sample2",
        "Prob097_mux9to1v__Gen2__Prob097_mux9to1v_M-E_sample5",
        "Prob105_rotate100__Gen0__Prob105_rotate100_initial_sample6",
        "Prob108_rule90__Gen1__Prob108_rule90_M-R_sample5",
        "Prob115_shift18__Gen2__Prob115_shift18_C-F_sample1",
        "Prob118_history_shift__Gen3__Prob118_history_shift_M-R_sample8",
        "Prob123_bugs_addsubz__Gen3__Prob123_bugs_addsubz_C-F_sample1"
    ]

    problems = [name.split("__")[0] for name in names]
    keys = ["eff_clk_period", "area", "power"]
    reference_ppas = {}
    for prob, name in zip(problems, names):
        data = pd.read_csv(f"data/bench/VerilogEval-Spec-to-RTL-DC-opt_resynth/{prob}_ppa.txt")
        ppa = {key: float(data[key].iloc[0]) for key in keys}
        reference_ppas[prob] = ppa
    print("Reference PPAs:")
    print(reference_ppas)
    best_ppas = {}
    best_scores = {}

    for data in ppa_feedback:
        problem_name = data["Name"].split("__")[0]

        name = data["Name"]
        synth = data["Syn"]
        result = None
        for synth_res in synth:
            if synth_res["Clock"] == clock:
                result = synth_res
                break
        if result is None:
            print(f"No result found for {name} at clock {clock}")
            continue
        ppa = {
            "eff_clk_period": result["Delay"],
            "area": result["Total Area"],
            "power": result["Total Power"]
        }
        score = fitness_score(ppa, reference_ppas[problem_name])
        if problem_name in best_ppas:
            if score > best_scores[problem_name]:
                best_ppas[problem_name] = ppa
                best_scores[problem_name] = score
        else:
            best_ppas[problem_name] = ppa
            best_scores[problem_name] = score

    for problem_name in problems:
        print("=" * 40)
        print(f"Comparing results for {problem_name}:")
        print(best_ppas[problem_name], reference_ppas[problem_name], best_scores[problem_name])
        print(f"Fitness Score: {best_scores[problem_name]:.4f}")

        # if data["Name"] in names:
        #     name = data["Name"]
        #     synth = data["Syn"]
        #     result = None
        #     for synth_res in synth:
        #         if synth_res["Clock"] == clock:
        #             result = synth_res
        #             break
        #     if result is None:
        #         print(f"No result found for {name} at clock {clock}")
        #         continue
        #     ppa = {
        #         "eff_clk_period": result["Delay"],
        #         "area": result["Total Area"],
        #         "power": result["Total Power"]
        #     }
        #     score = fitness_score(ppa, reference_ppas[name])
        #     print("=" * 40)
        #     print(f"Comparing results for {name}:")
        #     print(ppa, reference_ppas[name], score)
        #     print(f"Fitness Score: {score:.4f}")

    print(f"mean fitness score: {sum(best_scores.values()) / len(best_scores):.4f}")