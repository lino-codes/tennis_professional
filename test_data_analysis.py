import pandas as pd
import numpy as np
import datetime
from collections import defaultdict
from test_df_helper import pandas_show_all

pandas_show_all()

# NOTE:
class TennisDataAnalysis():
    def __init__(self):
        self.today_str = datetime.date.today().strftime('%Y%m%d')
        self.tennis_data = {'atp': pd.read_excel(f'./data/atp_tennis-data_{self.today_str}.xlsx'),
                            'wta': pd.read_excel(f'./data/wta_tennis-data_{self.today_str}.xlsx')}

        # NOTE: This is
        self.tennis_fy_data = {'atp': pd.read_excel(f'./data/atp_tennis-data_20251231.xlsx'),
                            'wta': pd.read_excel(f'./data/wta_tennis-data_20251231.xlsx')}

    def elo_construct(self):
        for tour_name, tour_df in self.tennis_fy_data.items():
            # NOTE: limiting the number of columns

            tour_df['Date'] = pd.to_datetime(tour_df['Date'])
            tour_df = tour_df.sort_values('Date').reset_index(drop=True)

            # Parameters
            INITIAL_ELO = 1500
            K = 32  # Adjust K as needed for tennis (20-40 common)
            match_count = defaultdict(int)
            elo_ratings = {}
            elo_count = {}

            def expected_score(r_a, r_b):
                """Expected win probability for A against B."""
                return 1 / (1 + 10 ** ((r_b - r_a) / 400))

            # Lists for new columns
            pre_winner_elo = []
            pre_loser_elo = []
            pre_winner_elo_count = []
            pre_loser_elo_count = []
            post_winner_elo = []
            post_loser_elo = []

            for _, row in tour_df.iterrows():
                winner = row['Winner']
                loser = row['Loser']

                # Initialize new players
                if winner not in elo_ratings:
                    elo_ratings[winner] = INITIAL_ELO
                    elo_count[winner] = 0
                if loser not in elo_ratings:
                    elo_ratings[loser] = INITIAL_ELO
                    elo_count[loser] = 0

                pre_w = elo_ratings[winner]
                pre_l = elo_ratings[loser]
                pre_winner_elo.append(pre_w)
                pre_loser_elo.append(pre_l)

                pre_w_count = elo_count[winner]
                pre_l_count = elo_count[loser]
                pre_winner_elo_count.append(pre_w_count)
                pre_loser_elo_count.append(pre_l_count)


                # Elo updates
                e_winner = expected_score(pre_w, pre_l)
                e_loser = expected_score(pre_l, pre_w)

                elo_ratings[winner] += K * (1 - e_winner)
                elo_ratings[loser] += K * (0 - e_loser)
                elo_count[winner] += 1
                elo_count[loser] += 1

                post_winner_elo.append(elo_ratings[winner])
                post_loser_elo.append(elo_ratings[loser])

            # Add columns
            tour_df['PreWElo'] = pre_winner_elo
            tour_df['PreLElo'] = pre_loser_elo
            tour_df['PostWElo'] = post_winner_elo
            tour_df['PostLElo'] = post_loser_elo
            tour_df['PreWEloCount'] = pre_winner_elo_count
            tour_df['PreLEloCount'] = pre_loser_elo_count
            print(tour_name)
            tour_df["HEloWins"] = np.where(tour_df["PreWElo"] > tour_df['PreLElo'], 1, 0)
            tour_df["LEloWins"] = np.where(tour_df["PreWElo"] < tour_df['PreLElo'], 1, 0)
            self.tennis_data[tour_name] = tour_df
            # print(tour_df)
            # rankings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
            # df_rank = pd.DataFrame(rankings, columns=['Player', 'Elo'])
            # df_rank['Rank'] = range(1, len(df_rank) + 1)
            # df_rank = df_rank[['Rank', 'Player', 'Elo']].round(2)
            # print(df_rank)


    def ranking_construct(self, strategy='HRankWins'):
        """Simple matches prediction methodology, either choosing the higher rank players would win,
        or the lower rank players would win. """
        for tour_name, tour_df in self.tennis_data.items():
            tour_df["HRankWins"] = np.where(tour_df["WRank"] < tour_df['LRank'], 1, 0)
            tour_df["LRankWins"] = np.where(tour_df["WRank"] > tour_df['LRank'], 1, 0)
            self.tennis_data[tour_name] = tour_df



    def ranking_strategy(self, strategy='HRankWins'):
        for tour_name, tour_df in self.tennis_data.items():
            tour_df["ranking_bet_odds"] = np.where((tour_df[strategy] == 1), tour_df["AvgW"], tour_df["AvgL"])

            tour_df["ranking_profit"] = np.where(tour_df[strategy], tour_df["ranking_bet_odds"] - 1, -1)
            total_profit = tour_df["ranking_profit"].sum()
            roi = total_profit / len(tour_df)
            print(f"Total ranking profit for {tour_name}: {total_profit}")

    def elo_strategy(self, strategy='HEloWins'):
        for tour_name, tour_df in self.tennis_data.items():
            tour_df["elo_bet_odds"] = np.where((tour_df[strategy] == 1), tour_df["AvgW"], tour_df["AvgL"])


            tour_df["elo_profit"] = np.where(tour_df[strategy], tour_df["elo_bet_odds"] - 1, -1)
            tour_df["elo_profit"] = np.where((tour_df['PreWEloCount'] > 2) & (tour_df['PreLEloCount'] > 2), tour_df["elo_profit"], 0)
            total_profit = tour_df["elo_profit"].sum()
            roi = total_profit / len(tour_df)
            print(tour_df[['Winner', 'Loser', 'PreWElo', 'PreLElo', 'PostWElo', 'PostLElo',  'PreWEloCount',
                           'PreLEloCount',  'HEloWins',  'LEloWins',  'elo_bet_odds',  'elo_profit']])
            print(f"Total elo profit for {tour_name}: {total_profit}")



    def run_analysis(self):
        self.ranking_construct()
        self.ranking_strategy()
        self.elo_construct()
        self.elo_strategy()
        # self.ranking_strategy(strategy='LRankWins')





