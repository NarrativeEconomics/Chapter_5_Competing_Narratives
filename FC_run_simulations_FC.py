import FC_predictionMarketsClasses_FC as pmcl_FC
from FC_predictionMarketsClasses_FC import opinion_diffusion
from FC_predictionMarketsClasses_FC import prediction_market


import numpy as np
import pickle
import sys
import pandas as pd


def FC_run_sim_FC(param):
    write_path = "C:/Users/oc21380/PycharmProjects/Chapter5_Competing_Narratives/Surrogate_Compare/Compare_BSE_WIth_surrogate/"


    alpha_p = param[0]  # 'alpha_p' list
    alpha_n = param[1]  # 'alpha_n' list




    N_simulations = 50
    N_agents = 100

    result_matrix =[]
    all_daily_opinions = []
    all_daily_normalized_opinions = []

    for sim in range(0,N_simulations):
        N_days = 119
        steps_per_day = 24*60  # or 1 if one loop is one day
        N_loops = N_days * steps_per_day

        print(f"Simulation {sim + 1}/{N_simulations}, Days: {N_days}, Loops: {N_loops}")

        traders = [pmcl_FC.agent() for _ in range(N_agents)]

        Negative_c = pmcl_FC.TradersGroup('Naive')
        Positive_c = pmcl_FC.TradersGroup('Rational')

        trader_list = list(traders)

        j = 1
        for trader in trader_list:
            if j % 2 == 0:
                Negative_c.add_trader(trader)
                trader.opinion = -0.3
                trader.alpha = 0.0
                trader.opinion_input = -0.05
                trader.gamma = 0.0
            else:
                Positive_c.add_trader(trader)
                trader.opinion = 0.3
                trader.alpha = 0.0
                trader.opinion_input = 0.05
                trader.gamma = 0.0
            j += 1

        op_param = [N_agents, N_days, traders, Negative_c, Positive_c]
        network = pmcl_FC.opinion_diffusion(op_param)

        pm_param = [0.24, 1]
        market = pmcl_FC.prediction_market(pm_param)

        daily_opinions = []
        daily_normalized_opinions = []

        for loop_index in range(N_loops):
            current_day = loop_index // steps_per_day


            for trader in trader_list:
                if trader in Negative_c.traders:
                    trader.alpha = alpha_n[current_day]
                    trader.opinion_input = -0.05

                else:
                    trader.alpha = alpha_p[current_day]
                    trader.opinion_input = 0.05

            network.curr_day = current_day
            traders = network.launch(traders) # updates opinons of 2 agets randomly selected (select one from each group)


            ag_id = network.update_op_series(loop_index, traders)
            market.launch(ag_id, network, traders)

            # Capture the opinions of all traders at the end of each "day"

            daily_opinions.append([trader.opinion  for trader in traders])

            daily_normalized_opinions.append([trader.normalize_opinion for trader in traders])


        all_daily_opinions.append(daily_opinions)
        all_daily_normalized_opinions.append(daily_normalized_opinions)
        # Append results for each simulation
        result_matrix.append(market.pt)

        # Save daily opinions to a file
        #daily_opinions_file = f"{write_path}Ch5_Surr_daily_opinions_{sim}_noise_SD_3.pickle"
        #daily_noramlized_opinions_file = f"{write_path}Ch5_Surr_daily_normalized_opinions_{sim}_noise_SD_.pickle"
        #with open(daily_opinions_file, "wb") as f:
        #    pickle.dump(all_daily_opinions, f)

        #with open(daily_noramlized_opinions_file, "wb") as f:
         #   pickle.dump(all_daily_normalized_opinions, f)

    # Save final market results as an matrix of all
    market_results_file = f"{write_path}Ch5_Surr_market_results_noise_SD_0.05.pickle"
    with open(market_results_file, "wb") as f:
        pickle.dump(result_matrix, f)
    return  result_matrix


def fC_main():



    alpha_df = pd.read_csv("alpha_values_5days.csv")

    alpha_p_values = alpha_df["alpha_p"].tolist()
    alpha_n_values = alpha_df["alpha_n"].tolist()


    param_PC = [alpha_p_values, alpha_n_values,]

    # Call the main function with the required parameters

    result_matrix = FC_run_sim_FC(param_PC)
    return result_matrix



if __name__ == "__main__":
    result_matrix = fC_main()
    print("Simulation finished.")






