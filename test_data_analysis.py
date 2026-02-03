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
        # NOTE: train data: referring to the full year training data
        self.train_data = {'atp': pd.read_excel(f'./data/atp_tennis-data_20251231.xlsx'),
                            'wta': pd.read_excel(f'./data/wta_tennis-data_20251231.xlsx')}

        # NOTE: test_data: data we are used to evaluate prediction strategy
        self.test_data = {'atp': pd.read_excel(f'./data/atp_tennis-data_{self.today_str}.xlsx'),
                            'wta': pd.read_excel(f'./data/wta_tennis-data_{self.today_str}.xlsx')}

        self.full_data = {
            'atp': pd.concat(
                [self.train_data['atp'], self.test_data['atp']],
                ignore_index=True
            ),
            'wta': pd.concat(
                [self.train_data['wta'], self.test_data['wta']],
                ignore_index=True
            ),
        }

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
        for tour_name, tour_df in self.full_data.items():
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
            self.full_data[tour_name] = tour_df


    def h2h_construct(self):
        '''
        Add pre-match head-to-head features to tennis match dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            Must contain columns: Date, Winner, Loser

        Returns:
        --------
        pd.DataFrame with additional columns:
            - Winner_H2H_Wins: Winner's wins vs Loser before this match
            - Winner_H2H_Losses: Winner's losses vs Loser before this match
            - Loser_H2H_Wins: Loser's wins vs Winner before this match
            - Loser_H2H_Losses: Loser's losses vs Winner before this match
        '''

        for tour_name, full_df in self.full_data.items():
            df = full_df.sort_values('Date').copy()

            df['Winner_H2H_Wins'] = 0
            df['Winner_H2H_Losses'] = 0
            df['Loser_H2H_Wins'] = 0
            df['Loser_H2H_Losses'] = 0

            match_history = {}

            for idx, row in df.iterrows():
                winner = row['Winner']
                loser = row['Loser']

                players = tuple(sorted([winner, loser]))

                if players in match_history:
                    prev_matches = match_history[players]
                    winner_prev_wins = sum(1 for _, w in prev_matches if w == winner)
                    loser_prev_wins = sum(1 for _, w in prev_matches if w == loser)

                    df.at[idx, 'Winner_H2H_Wins'] = winner_prev_wins
                    df.at[idx, 'Winner_H2H_Losses'] = loser_prev_wins
                    df.at[idx, 'Loser_H2H_Wins'] = loser_prev_wins
                    df.at[idx, 'Loser_H2H_Losses'] = winner_prev_wins

                if players not in match_history:
                    match_history[players] = []
                match_history[players].append((row['Date'], winner))
            self.full_data[tour_name] = df


    def h2h_feature_engineering(self):
        for tour_name, full_df in self.full_data.items():
            df = full_df.copy()
            # H2H features
            df['h2h_total'] = df['Winner_H2H_Wins'] + df['Winner_H2H_Losses']
            df['h2h_win_rate'] = np.where(df['h2h_total'] > 0,
                                          df['Winner_H2H_Wins'] / df['h2h_total'], 0.5)
            df['h2h_advantage'] = df['Winner_H2H_Wins'] - df['Winner_H2H_Losses']

            print('We are in h2h feature engineering')
            # NOTE: The PTS DIFF

            # print(df[['Date', 'Winner', 'Loser', 'Winner_H2H_Wins', 'Loser_H2H_Wins', 'h2h_total', 'h2h_win_rate',
            #           'h2h_advantage', 'AbsPtsDiff', 'PtsDiff', 'HRankWins','WPts', 'LPts', 'WPTs_Ratio']].tail(10))

            df = df.dropna(subset=['WRank', 'LRank'])
            X = df[['Pts_Ratio', 'h2h_win_rate']]
            y = df['HRankWins']  # 1 if higher-ranked player won
            model = LogisticRegression()

            split_idx = int(len(df) * 0.8)

            X_train = X[:split_idx]
            X_test = X[split_idx:]
            y_train = y[:split_idx]
            y_test = y[split_idx:]


            # Train-test split (80/20)
            model.fit(X_train, y_train)
            X_test = df[['Pts_Ratio', 'h2h_win_rate']]
            df['Prob_HRankWins'] = model.predict_proba(X_test)[:, 1]
            print('h2h feature engineering')
            print(df.sort_values(by=['Pts_Ratio']).tail())


            # # Ranking features
            # df['rank_diff'] = df['WRank'] - df['LRank']  # Negative = winner is higher ranked
            # df['rank_ratio'] = df['LRank'] / (df['WRank'] + 1e-6)
            # df['pts_diff'] = df['WPts'] - df['LPts']
            # df['pts_ratio'] = df['WPts'] / (df['LPts'] + 1e-6)
            #
            # # Interaction features
            # df['h2h_x_rank'] = df['h2h_advantage'] * df['rank_diff']
            # df['h2h_x_pts'] = df['h2h_advantage'] * df['pts_diff'] / 1000






    def ranking_construct(self, strategy='HRankWins'):
        """Simple matches prediction methodology, either choosing the higher rank players would win,
        or the lower rank players would win. """
        for tour_name, test_df in self.test_data.items():
            test_df["HRankWins"] = np.where(test_df["WRank"] < test_df['LRank'], 1, 0)
            test_df["LRankWins"] = np.where(test_df["WRank"] > test_df['LRank'], 1, 0)
            test_df['RankDiff'] = test_df['WRank'] - test_df['LRank']
            test_df['PtsDiff'] = test_df['WPts'] - test_df['LPts']
            test_df['AbsPtsDiff'] = test_df['PtsDiff'].abs()
            test_df['Pts_Ratio'] = test_df[['WPts', 'LPts']].max(axis=1) / (test_df['WPts'] + test_df['LPts'])
            self.test_data[tour_name] = test_df

        for tour_name, train_df in self.train_data.items():
            train_df["HRankWins"] = np.where(train_df["WRank"] < train_df['LRank'], 1, 0)
            train_df["LRankWins"] = np.where(train_df["WRank"] > train_df['LRank'], 1, 0)
            train_df['RankDiff'] = train_df['WRank'] - train_df['LRank']
            train_df['PtsDiff'] = train_df['WPts'] - train_df['LPts']
            train_df['AbsPtsDiff'] = train_df['PtsDiff'].abs()
            train_df['Pts_Ratio'] = train_df[['WPts', 'LPts']].max(axis=1) / (train_df['WPts'] + train_df['LPts'])
            self.train_data[tour_name] = train_df


        # NOTE: If using full, you must create a train test split
        for tour_name, full_df in self.full_data.items():
            full_df["HRankWins"] = np.where(full_df["WRank"] < full_df['LRank'], 1, 0)
            full_df["LRankWins"] = np.where(full_df["WRank"] > full_df['LRank'], 1, 0)
            full_df['RankDiff'] = full_df['WRank'] - full_df['LRank']
            full_df['PtsDiff'] = full_df['WPts'] - full_df['LPts']
            full_df['AbsPtsDiff'] = full_df['PtsDiff'].abs()
            full_df['Pts_Ratio'] = full_df[['WPts', 'LPts']].max(axis=1) / (full_df['WPts'] + full_df['LPts'])
            self.full_data[tour_name] = full_df



    def ranking_stats_construct(self, strategy='HRankWins'):
        """The accuracy and the profit we get if we use the ranking strategy. ranking construct must be run
        Profit: The potential profit after adopting the
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
         THIS IS JUST STATS, SO NO NEED TO DO TRAIN TEST SPLIT"""

        for tour_name, test_df in self.test_data.items():
            test_df["ranking_bet_odds"] = np.where((test_df[strategy] == 1), test_df["AvgW"], test_df["AvgL"])
            test_df["ranking_profit"] = np.where(test_df[strategy], test_df["ranking_bet_odds"] - 1, -1)
            total_profit = test_df["ranking_profit"].sum()
            rank_accuracy = test_df[strategy].mean() * 100
            print(f'The potential profit for {tour_name} using {strategy} in TEST DATA is {total_profit:.2f}')
            print(f'The accuracy for {tour_name} using {strategy} is {rank_accuracy:.2f}%')
            corr_rank = scipy.stats.pointbiserialr(test_df['HRankWins'], test_df['RankDiff'])
            test_df = test_df.dropna(subset=['WRank', 'LRank'])
            print(f"{tour_name} for TEST DATA: {strategy} vs RankDiff: strength, r = {corr_rank[0]:.4f}, "
                  f"variance, r2 = {corr_rank[0]**2:.4f}, "
                  f"p-value = {corr_rank[1]:.4f}")

            # Correlation 2: HRankWins vs PtsDiff
            corr_pts = scipy.stats.pointbiserialr(test_df['HRankWins'], test_df['PtsDiff'])
            print(f"{tour_name} for TEST DATA: {strategy} vs PtsDiff: strength, r = {corr_pts[0]:.4f},"
                  f"variance, r2 =  {corr_pts[0]**2:.4f}, "
                  f"p = {corr_pts[1]:.4f}")
            self.test_data[tour_name] = test_df

        for tour_name, train_df in self.train_data.items():
            train_df["ranking_bet_odds"] = np.where((train_df[strategy] == 1), train_df["AvgW"], train_df["AvgL"])
            train_df["ranking_profit"] = np.where(train_df[strategy], train_df["ranking_bet_odds"] - 1, -1)

            total_profit = train_df["ranking_profit"].sum()
            rank_accuracy = train_df[strategy].mean() * 100
            print(f'The potential profit for {tour_name} using {strategy} in TRAIN DATA is {total_profit:.2f}')
            print(f'The accuracy for {tour_name} using {strategy} is {rank_accuracy:.2f}%')
            train_df = train_df.dropna(subset=['WRank', 'LRank'])
            corr_rank = scipy.stats.pointbiserialr(train_df['HRankWins'], train_df['RankDiff'])
            print(f"{tour_name} for TRAIN DATA: {strategy} vs RankDiff: strength, r = {corr_rank[0]:.4f}, "
                  f"variance, r2 = {corr_rank[0]**2:.4f}, "
                  f"p-value = {corr_rank[1]:.4f}")

            # Correlation 2: HRankWins vs PtsDiff
            corr_pts = scipy.stats.pointbiserialr(train_df['HRankWins'], train_df['PtsDiff'])
            print(f"{tour_name} for TRAIN DATA: {strategy} vs PtsDiff: strength, r = {corr_pts[0]:.4f},"
                  f"variance, r2 =  {corr_pts[0]**2:.4f}, "
                  f"p = {corr_pts[1]:.4f}")
            self.train_data[tour_name] = train_df


        for tour_name, full_df in self.full_data.items():
            full_df["ranking_bet_odds"] = np.where((full_df[strategy] == 1), full_df["AvgW"], full_df["AvgL"])
            full_df["ranking_profit"] = np.where(full_df[strategy], full_df["ranking_bet_odds"] - 1, -1)

            total_profit = full_df["ranking_profit"].sum()
            rank_accuracy = full_df[strategy].mean() * 100
            print(f'The potential profit for {tour_name} using {strategy} in TRAIN DATA is {total_profit:.2f}')
            print(f'The accuracy for {tour_name} using {strategy} is {rank_accuracy:.2f}%')
            full_df = full_df.dropna(subset=['WRank', 'LRank'])
            corr_rank = scipy.stats.pointbiserialr(full_df['HRankWins'], full_df['RankDiff'])
            print(f"{tour_name} for FULL DATA: {strategy} vs RankDiff: strength, r = {corr_rank[0]:.4f}, "
                  f"variance, r2 = {corr_rank[0]**2:.4f}, "
                  f"p-value = {corr_rank[1]:.4f}")

            # Correlation 2: HRankWins vs PtsDiff
            corr_pts = scipy.stats.pointbiserialr(full_df['HRankWins'], full_df['PtsDiff'])
            print(f"{tour_name} for FULL DATA: {strategy} vs PtsDiff: strength, r = {corr_pts[0]:.4f},"
                  f"variance, r2 =  {corr_pts[0]**2:.4f}, "
                  f"p = {corr_pts[1]:.4f}")
            self.full_data[tour_name] = full_df


    def model_fitting(self, feature_cols, target_variable, confidence_level=0.6):
        # NOTE: This is using the full year data as training data,
        print('We are in the model fitting phase')
        for tour_name, train_df in self.train_data.items():
            X = train_df[feature_cols]
            y = train_df[target_variable]
            model = LogisticRegression()
            model.fit(X, y)
            # NOTE: After fitting the model, now using the model to predict
            test_df = self.test_data[tour_name]
            X_test = test_df[feature_cols]
            test_df['Prob_HRankWins'] = model.predict_proba(X_test)[:, 1]  # after fitting full year data
            confident_df = test_df[test_df['Prob_HRankWins'] > confidence_level]
            # print(confident_df[['Winner', 'Loser', 'HRankWins',  'WPts', 'LPts', 'PtsDiff',
            #                'Prob_HRankWins', 'ranking_bet_odds', 'ranking_profit']].sort_values(by='Prob_HRankWins', ascending=False))

            total_profit = confident_df["ranking_profit"].sum()
            rank_accuracy = confident_df['HRankWins'].mean() * 100
            profit_per_bet = total_profit / len(confident_df)
            # print(confident_df.reset_index(drop=True))
            print(confident_df[['Date', 'Winner', 'Loser']].head())
            print(f'Based on a confidence level of {confidence_level}')
            print(f'The potential profit for {tour_name} using {'HRankWins'} is {total_profit:.2f}')
            print(f'The accuracy for {tour_name} using {'HrankWins'} is {rank_accuracy:.2f}%')
            print(f'The average profit per bet for {tour_name} using {'HrankWins'} is {profit_per_bet:.2f}')

        # NOTE: End of data





    def elo_strategy(self, strategy='HEloWins'):
        for tour_name, tour_df in self.tennis_data.items():
            tour_df["elo_bet_odds"] = np.where((tour_df[strategy] == 1), tour_df["AvgW"], tour_df["AvgL"])
            tour_df["elo_profit"] = np.where(tour_df[strategy], tour_df["elo_bet_odds"] - 1, -1)
            tour_df["elo_profit"] = np.where((tour_df['PreWEloCount'] > 12) & (tour_df['PreLEloCount'] > 12), tour_df["elo_profit"], 0)
            total_profit = tour_df["elo_profit"].sum()
            roi = total_profit / len(tour_df)
            # print(tour_df[['Winner', 'Loser', 'PreWElo', 'PreLElo', 'PostWElo', 'PostLElo',  'PreWEloCount',
            #                'PreLEloCount',  'HEloWins',  'LEloWins',  'elo_bet_odds',  'elo_profit']])
            print(f"Total elo profit for {tour_name}: {total_profit}")\

    def run_analysis(self):
        self.h2h_construct()

        self.ranking_construct()
        self.h2h_feature_engineering()
        self.ranking_stats_construct()
        self.model_fitting(feature_cols=['Pts_Ratio'], target_variable='HRankWins', confidence_level=0.65)
        # self.elo_construct(elo_start_date=datetime.date(2024,12,20))
        # self.elo_strategy()





