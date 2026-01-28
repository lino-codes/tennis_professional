import pandas as pd
import numpy as np
import datetime
import scipy
from scipy.stats import spearmanr
from pathlib import Path
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split

from test_df_helper import pandas_show_all
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


pandas_show_all()

# NOTE:
class TennisDataAnalysis():
    def __init__(self):
        self.today_str = datetime.date.today().strftime('%Y%m%d')
        self.tennis_data = {'atp': pd.read_excel(f'./data/atp_tennis-data_{self.today_str}.xlsx'),
                            'wta': pd.read_excel(f'./data/wta_tennis-data_{self.today_str}.xlsx')}

        # NOTE: test data refers to data that will be used for prediction
        self.test_data = {'atp': pd.read_excel(f'./data/atp_tennis-data_20251231.xlsx'),
                            'wta': pd.read_excel(f'./data/wta_tennis-data_20251231.xlsx')}
        # self.test_data = {'atp': pd.read_excel(f'./data/atp_tennis-data_{self.today_str}.xlsx'),
        #                     'wta': pd.read_excel(f'./data/wta_tennis-data_{self.today_str}.xlsx')}

    def elo_construct(self, elo_start_date=datetime.date.today() - relativedelta(years=1), initial_elo=1500, k=40):
        """Elo Construction based on Own Criteria.
         elo_start_date:
            we already have past full year matches data, indicate when you want to elo to be calculated.
         """

        # NOTE: getting the elo df based on full year data and current year data
        folder = Path("./data")
        data_files = list(folder.glob("*tennis-data*.xlsx"))
        def year_from_path(p: Path) -> int:
            # filename like atp_tennis-data_20241231.xlsx
            date_str = p.stem.split("_")[-1]  # "20241231"
            return int(date_str[:4])  # 2024

        full_year_data = [
            p for p in data_files
            if p.name.endswith("1231.xlsx") and year_from_path(p) >= elo_start_date.year
        ]

        def file_date(p: Path) -> datetime.date:
            # stem: "atp_tennis-data_20241231"
            stem = p.stem
            ds = stem.split("_")[-1]  # "20241231"
            return datetime.date(int(ds[:4]), int(ds[4:6]), int(ds[6:]))

        # filter files strictly before cutoff date

        atp_files = [p for p in full_year_data if "atp_" in p.name]
        wta_files = [p for p in full_year_data if "wta_" in p.name]

        df_atp = pd.concat((pd.read_excel(p) for p in atp_files), ignore_index=True)
        df_wta = pd.concat((pd.read_excel(p) for p in wta_files), ignore_index=True)
        df_atp = df_atp[df_atp["Date"] >= pd.Timestamp(elo_start_date)]
        df_wta = df_wta[df_wta['Date'] >= pd.Timestamp(elo_start_date)]
        for tour_name, tour_df in self.tennis_data.items():
            if tour_name == "atp":
                tour_df = pd.concat([tour_df, df_atp], ignore_index=True)
            else:
                tour_df = pd.concat([tour_df, df_wta], ignore_index=True)
            tour_df = tour_df.sort_values(by=['Date'])

            # NOTE: This starts the Elo calculation
            elo_ratings = {}
            elo_count = {}

            def expected_score(r_a, r_b):
                """Expected win probability for A against B."""
                return 1 / (1 + 10 ** ((r_b - r_a) / 400))

            # creating new columns
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
                    elo_ratings[winner] = initial_elo
                    elo_count[winner] = 0
                if loser not in elo_ratings:
                    elo_ratings[loser] = initial_elo
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

                elo_ratings[winner] += k * (1 - e_winner)
                elo_ratings[loser] += k * (0 - e_loser)
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

            tour_df["HEloWins"] = np.where(tour_df["PreLElo"] < tour_df['PreWElo'], 1, 0)
            tour_df["LEloWins"] = np.where(tour_df["PreLElo"] > tour_df['PreWElo'], 1, 0)
            rankings = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
            df_rank = pd.DataFrame(rankings, columns=['Player', 'Elo'])
            df_rank['Rank'] = range(1, len(df_rank) + 1)
            df_rank = df_rank[['Rank', 'Player', 'Elo']].round(2)

            df_rank['MatchCount'] = df_rank['Player'].map(elo_count)
            df_rank = df_rank[df_rank['MatchCount'] >= 12]
            df_rank['Rank'] = range(1, len(df_rank) + 1)
            df_rank = df_rank.set_index('Rank')
            print('Top Elo players:', df_rank.head(20))
            self.tennis_data[tour_name] = tour_df

    def ranking_construct(self, strategy='HRankWins'):
        """Simple matches prediction methodology, either choosing the higher rank players would win,
        or the lower rank players would win. """
        for tour_name, tour_df in self.test_data.items():
            tour_df["HRankWins"] = np.where(tour_df["WRank"] < tour_df['LRank'], 1, 0)
            tour_df["LRankWins"] = np.where(tour_df["WRank"] > tour_df['LRank'], 1, 0)
            self.test_data[tour_name] = tour_df



    def ranking_strategy(self, strategy='HRankWins'):
        """Profit: The potential profit after adopting the
         Evaluation:
         Strength (r) measures the strength and direction of a linear relationship between two variables
         ranges from -1 to +1:
         0.0 - 0.3: Weak correlation
         0.3 - 0.5: Moderate correlation
         0.5 - 0.7: Moderately strong correlation
         0.7 - 0.9: Strong correlation
         0.9 - 1.0: Very strong correlation
         variance = r**2
         Variance tells you the proportion of variation in one variable that can be predicted from other variable.
         33.7% of the variation in match outcomes is due to points difference, while 66.3% is due to other factors.
         p-value: how confident you can be that the relationship is real.
         """
        for tour_name, tour_df in self.test_data.items():
            print(f'The date range of data is from {tour_df['Date'].min().strftime("%Y%m%d")} to '
                  f'{tour_df['Date'].max().strftime("%Y%m%d")}')
            # print(tour_df[['Winner', 'Loser', 'WRank', 'LRank', 'WPts', 'LPts', strategy]].head())
            # NOTE: removing matches if the players have no ranking
            tour_df = tour_df.dropna(subset=['WRank', 'LRank'])
            # NOTE: End of

            # NOTE: Implication of adopting the strategy
            tour_df["ranking_bet_odds"] = np.where((tour_df[strategy] == 1), tour_df["AvgW"], tour_df["AvgL"])
            tour_df["ranking_profit"] = np.where(tour_df[strategy], tour_df["ranking_bet_odds"] - 1, -1)
            total_profit = tour_df["ranking_profit"].sum()
            rank_accuracy = tour_df[strategy].mean()*100
            print(f'The potential profit for {tour_name} using {strategy} is {total_profit:.2f}')
            print(f'The accuracy for {tour_name} using {strategy} is {rank_accuracy:.2f}%')
            # NOTE: End of

            tour_df['RankDiff'] = tour_df['WRank'] - tour_df['LRank']
            tour_df['PtsDiff'] = tour_df['WPts'] - tour_df['LPts']
            print(tour_df[['Winner', 'Loser', strategy, 'RankDiff', 'PtsDiff']].tail())
            # Correlation 1: HRankWins vs RankDiff
            corr_rank = scipy.stats.pointbiserialr(tour_df['HRankWins'], tour_df['RankDiff'])
            print(f"{tour_name}: HRankWins vs RankDiff: strength, r = {corr_rank[0]:.4f}, "
                  f"variance, r2 = {corr_rank[0]**2:.4f}, "
                  f"p-value = {corr_rank[1]:.4f}")

            # Correlation 2: HRankWins vs PtsDiff
            corr_pts = scipy.stats.pointbiserialr(tour_df['HRankWins'], tour_df['PtsDiff'])
            print(f"{tour_name}: HRankWins vs PtsDiff: strength, r = {corr_pts[0]:.4f},"
                  f"variance, r2 =  {corr_pts[0]**2:.4f}, "
                  f"p = {corr_pts[1]:.4f}")

            # NOTE: Check whether points difference is a good way to predict
            tour_df['AbsPtsDiff'] = tour_df['PtsDiff'].abs()
            X = tour_df[['AbsPtsDiff']]
            y = tour_df['HRankWins']
            model = LogisticRegression()
            model.fit(X, y)

            tour_df['Prob_HRankWins_1'] = model.predict_proba(X)[:, 1]
            print(tour_df[['Winner', 'Loser', 'HRankWins',  'WPts', 'LPts', 'PtsDiff',
                           'Prob_HRankWins_1']].sort_values(by='Prob_HRankWins_1', ascending=False).head(10))









    def elo_strategy(self, strategy='HEloWins'):
        for tour_name, tour_df in self.tennis_data.items():
            tour_df["elo_bet_odds"] = np.where((tour_df[strategy] == 1), tour_df["AvgW"], tour_df["AvgL"])
            tour_df["elo_profit"] = np.where(tour_df[strategy], tour_df["elo_bet_odds"] - 1, -1)
            tour_df["elo_profit"] = np.where((tour_df['PreWEloCount'] > 12) & (tour_df['PreLEloCount'] > 12), tour_df["elo_profit"], 0)
            total_profit = tour_df["elo_profit"].sum()
            roi = total_profit / len(tour_df)
            # print(tour_df[['Winner', 'Loser', 'PreWElo', 'PreLElo', 'PostWElo', 'PostLElo',  'PreWEloCount',
            #                'PreLEloCount',  'HEloWins',  'LEloWins',  'elo_bet_odds',  'elo_profit']])
            print(f"Total elo profit for {tour_name}: {total_profit}")


    def strategy_evaluation(self):
        print('Evaluating strategy')
        for tour_name, tour_df in self.tennis_data.items():
            print(tour_name)
            print(tour_df)
            df = tour_df[['Winner', 'Loser',
                          # 'PreWElo', 'PreLElo', 'HEloWins', 'LEloWins',
                          'HRankWins', 'LRankWins',
                          'WRank', 'LRank']]
            print(df.tail())
            rank_accuracy = df['HRankWins'].mean()
            print(rank_accuracy)
            # elo_accuracy = df['HEloWins'].mean()
            # print(elo_accuracy)




    def run_analysis(self):
        self.ranking_construct()
        self.ranking_strategy()
        # self.elo_construct(elo_start_date=datetime.date(2026,1,1))
        # self.elo_strategy()
        # self.strategy_evaluation()

        # self.ranking_strategy(strategy='LRankWins')





