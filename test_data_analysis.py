import pandas as pd
import numpy as np
import datetime
import scipy
from scipy.stats import spearmanr
from pathlib import Path
from collections import defaultdict
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split

from constants import relevant_columns
from test_df_helper import pandas_show_all
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


pandas_show_all()

# TODO: Clean up function one by one


class TennisDataAnalysis():
    def __init__(self):
        self.today_str = datetime.date.today().strftime('%Y%m%d')
        # NOTE: train data: referring to the full year training data
        # self.train_data = {'atp': pd.read_excel(f'./data/atp_tennis-data_20251231.xlsx'),
        #                     'wta': pd.read_excel(f'./data/wta_tennis-data_20251231.xlsx')}

        self.train_data = {'atp': pd.read_excel(f'./data/atp_tennis-data_2024-2025.xlsx', index_col=0),
                            'wta': pd.read_excel(f'./data/wta_tennis-data_2024-2025.xlsx', index_col=0)}


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

        # NOTE: Temporary data for better visualisation
        # self.temp_data = {'atp': self.full_data['atp'][relevant_columns],
        #                   'wta': self.full_data['wta'][relevant_columns]}
        self.temp_data = {'atp': self.full_data['atp'],
                          'wta': self.full_data['wta']}

    def elo_construct(self, elo_start_date=datetime.date.today() - relativedelta(years=1), initial_elo=1500, k=40):
        # NOTE: We are not currently looking at Elo
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


    def get_rank_feature(self):
        print('GET Rank Features')
        for tour_name, temp_df in self.full_data.items():
            for col in ["WRank", "LRank", "WPts", "LPts"]:
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')

            temp_df = temp_df.dropna(subset=['WRank', 'LRank', 'WPts', 'LPts']) # drop if any relevant role is nan
            print(temp_df.columns)
            # replace any potential zeros to avoid log issues
            temp_df["WRank"] = temp_df["WRank"].clip(lower=1.0)
            temp_df["LRank"] = temp_df["LRank"].clip(lower=1.0)
            temp_df["WPts"] = temp_df["WPts"].clip(lower=0.0)
            temp_df["LPts"] = temp_df["LPts"].clip(lower=0.0)
            # NOTE: End of dataframe cleaning
            # NOTE: NOW applying log
            def build_pairwise_rows(row):
                a1 = -np.log(row["WRank"]) # NOTE: Unsure what these are
                b1 = np.log(row["WPts"] + 1.0)
                a2 = -np.log(row["LRank"])
                b2 = np.log(row["LPts"] + 1.0)

                r_win = {
                    "a_rank_logneg": a1,
                    "a_pts_log": b1,
                    "b_rank_logneg": a2,
                    "b_pts_log": b2,
                    "y": 1
                }

                r_lose = {
                    "a_rank_logneg": a2,
                    "a_pts_log": b2,
                    "b_rank_logneg": a1,
                    "b_pts_log": b1,
                    "y": 0
                }
                return pd.DataFrame([r_win, r_lose])
            train_rows = []
            for _, row in temp_df.iterrows():
                train_rows.append(build_pairwise_rows(row)) # each pair is the same match, one is a winning row,
                # the other is losing row

            train_df = pd.concat(train_rows, ignore_index=True)

            # Feature matrix: we'll let the model learn the combination:
            # rank_feature_A - rank_feature_B is effectively a linear combination if we use:
            # X = [a_rank_logneg - b_rank_logneg, a_pts_log - b_pts_log]

            train_df["d_ranklog"] = train_df["a_rank_logneg"] - train_df["b_rank_logneg"]
            train_df["d_ptslog"] = train_df["a_pts_log"] - train_df["b_pts_log"]

            X = train_df[["d_ranklog", "d_ptslog"]].values
            y = train_df["y"].values

            # Fitting a calibrated logistic model

            pipe = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("logreg", LogisticRegression(solver="lbfgs"))
            ])
            # NOTE: Step 1: fits on training data to compute per feature mean and std,
            #  transforms in puts to zero mean, unit variance so features are on comparable scales
            # NOTE: Step 2: logistics regression trains a logistic regression classifier on the scaled features
            # Uses l2 regularisation by default, lbfgs is a stable,

            pipe.fit(X, y)

            # ------------------------------------------------------------
            d_ranklog = -np.log(temp_df["WRank"]) - (-np.log(temp_df["LRank"]))  # = -log(WRank) + log(LRank)
            d_ptslog = np.log(temp_df["WPts"] + 1.0) - np.log(temp_df["LPts"] + 1.0)

            X_eval = np.c_[d_ranklog.values, d_ptslog.values]
            p_winner = pipe.predict_proba(X_eval)[:, 1]  # probability A (winner) wins

            temp_df["p_rank_feature_winner"] = p_winner # predicted probability of the winning player winning

            self.full_data[tour_name] = temp_df


    def h2h_get(self):
        # NOTE: Currently using
        """
        Add pre-match head-to-head features to tennis match dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            Must contain columns: Date, Winner, Loser

        Returns:
        --------
        pd.DataFrame with additional columns:
            - Winner_H2H_Wins: Winner's wins vs Loser before this match
            - Loser_H2H_Wins: Loser's wins vs Winner before this match
        """

        for tour_name, full_df in self.full_data.items():
            df = full_df.sort_values('Date').copy()

            df['Winner_H2H_Wins'] = 0
            df['Loser_H2H_Wins'] = 0

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
                    df.at[idx, 'Loser_H2H_Wins'] = loser_prev_wins

                if players not in match_history:
                    match_history[players] = []
                match_history[players].append((row['Date'], winner))

            self.full_data[tour_name] = df

    def h2h_feature(self):
        # NOTE: This is to create feature that's relevant to h2h
        print('We are now in h2h_feature')
        # NOTE: New feature: I should create a timed adjused h2h

        for tour_name, full_df in self.full_data.items():
            print(tour_name)
            df = full_df.copy()
            df['h2h_matches'] = df['Winner_H2H_Wins'].fillna(0) + df['Loser_H2H_Wins'].fillna(0)
            alpha = 1.0  # Pseudocount (standard Laplace)
            # Smoothed win share
            df['h2h_win_share_sm'] = (df['Winner_H2H_Wins'] + alpha) / (df['h2h_matches'] + 2 * alpha)

            # NOTE: Clip extremes: for <3 matches, pull hard to 0.5 (optional but recommended)
            # mask_small = df['h2h_matches'] < 3
            # df.loc[mask_small, 'h2h_win_share_sm'] = np.clip(
            #     df.loc[mask_small, 'h2h_win_share_sm'], 0.4, 0.6
            # )
            self.full_data[tour_name] = df


    def rank_feature_validation(self):
        for tour_name, temp_df in self.full_data.items():
            evaluate_df = temp_df.copy()
            # evaluate_df["expected_gain"] = np.where(evaluate_df['p_winner'] > 0.5, evaluate_df["AvgW"] - 1, -1)

            # NOTE: Check if the probability beats the Bet365 Odds and set Confidence level, the higher it is, the bigger the discrepancy between our prediction and betting odds
            disc_level = 0.2
            realistic_level = 5
            book_fav = 'B365'

            # these are inplace to avoid errors
            evaluate_df = evaluate_df[~evaluate_df[f"{book_fav}W"].apply(lambda x: isinstance(x, str))]
            evaluate_df = evaluate_df[evaluate_df[f'{book_fav}W'] > 0]

            evaluate_df["implied_W"] = 1 / evaluate_df[f"{book_fav}W"] # the implied probability of winning
            evaluate_df["implied_L"] = 1 / evaluate_df[f"{book_fav}L"]

            evaluate_df["p_rank_feature_loser"] = 1 - evaluate_df["p_rank_feature_winner"] # predicted probability of the winning player winning

            self.full_data[tour_name] = evaluate_df
            # NOTE: To spot the betting opportunity
            evaluate_df["winner_margin"] = evaluate_df["p_rank_feature_winner"] - evaluate_df["implied_W"]
            evaluate_df["loser_margin"] = evaluate_df["p_rank_feature_loser"] - evaluate_df["implied_L"]

            winning_df = evaluate_df[(evaluate_df["winner_margin"] > disc_level)
                                  & (evaluate_df[f"{book_fav}W"] < realistic_level)]

            losing_df = evaluate_df[(evaluate_df["loser_margin"] > disc_level)
                                  & (evaluate_df[f"{book_fav}L"] < realistic_level)]


            winning_df['winning_money'] = winning_df[f'{book_fav}W'] - 1
            print(f'Tour Name: {tour_name}')
            print(f'Winning Money: {winning_df["winning_money"].sum()}')
            print(f'Losing Money: {len(losing_df)}')
            # print(winning_df.sort_values(f'{book_fav}W', ascending=False)[relevant_columns].head(15))
            # print(winning_df.sort_values(f'{book_fav}W', ascending=False).head())

            # print(evaluate_df[(evaluate_df["EV_winner"] > disc_level)].head())
            # print(evaluate_df[(evaluate_df["EV_loser"] > disc_level)].head())



    def ranking_get(self):
        # NOTE: This should be self explanator
        for tour_name, full_df in self.full_data.items():
            full_df["HRankWins"] = np.where(full_df["WRank"] < full_df['LRank'], 1, 0)
            full_df["LRankWins"] = np.where(full_df["WRank"] > full_df['LRank'], 1, 0)
            full_df['RankDiff'] = full_df['WRank'] - full_df['LRank']
            full_df['PtsDiff'] = full_df['WPts'] - full_df['LPts']
            full_df['AbsPtsDiff'] = full_df['PtsDiff'].abs()
            full_df['Pts_Ratio'] = full_df[['WPts', 'LPts']].max(axis=1) / (full_df['WPts'] + full_df['LPts'])
            self.full_data[tour_name] = full_df


    def ranking_stats(self, strategy='HRankWins',
                      start_date=datetime.datetime(datetime.datetime.today().year, 1, 1),
                      end_date=datetime.datetime.today()):
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
        for tour_name, full_df in self.full_data.items():
            full_df["ranking_bet_odds"] = np.where((full_df[strategy] == 1), full_df["AvgW"], full_df["AvgL"])
            full_df["ranking_profit"] = np.where(full_df[strategy], full_df["ranking_bet_odds"] - 1, -1)

            # print(full_df.head())
            df = full_df[(full_df['Date'] >= start_date) & (full_df['Date'] <= end_date)]

            total_profit = df["ranking_profit"].sum()
            rank_accuracy = df[strategy].mean() * 100
            print(f'For ranking statistics, only looking at dates from {start_date.strftime("%Y%m%d")} to {end_date.strftime("%Y%m%d")}')
            print(f'The potential profit for {tour_name} using {strategy} is {total_profit:.2f}')
            print(f'The accuracy for {tour_name} using {strategy} is {rank_accuracy:.2f}%')
            df = df.dropna(subset=['WRank', 'LRank'])
            corr_rank = scipy.stats.pointbiserialr(df['HRankWins'], df['RankDiff'])
            print(f"{tour_name} for FULL DATA: {strategy} vs RankDiff: strength, r = {corr_rank[0]:.4f}, "
                  f"variance, r2 = {corr_rank[0]**2:.4f}, "
                  f"p-value = {corr_rank[1]:.4f}")

            # Correlation 2: HRankWins vs PtsDiff
            corr_pts = scipy.stats.pointbiserialr(df['HRankWins'], df['PtsDiff'])
            print(f"{tour_name} for FULL DATA: {strategy} vs PtsDiff: strength, r = {corr_pts[0]:.4f},"
                  f"variance, r2 =  {corr_pts[0]**2:.4f}, "
                  f"p = {corr_pts[1]:.4f}")
            self.full_data[tour_name] = full_df


    def model_evaluation(self):
        """"""
        for tour_name, full_df in self.full_data.items():
            print(tour_name)
            # NOTE: Just having a look at what's insdie the dataframe
            # relevant_cols = ['Winner', 'Loser', 'WPts', 'LPts', 'Pts_Ratio', 'h2h_win_share_sm']
            # part_df = full_df[relevant_cols].tail(30)
            # print(part_df)
            # NOTE: End of
            # NOTE: Some dataprep
            # full_df['WPtsRatio'] = full_df['WPts'] / (full_df['WPts'] + full_df['LPts'])
            X = full_df[['AbsPtsDiff', 'h2h_win_share_sm']]
            Y = full_df['HRankWins']
            # NOTE: Tree/Ensemble Implementation
            split_idx = int(0.8*len(full_df))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            Y_train, Y_test = Y.iloc[:split_idx], Y.iloc[split_idx:]
            xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                                eval_metric='logloss')
            xgb.fit(X_train, Y_train)
            xgb_proba = xgb.predict_proba(X_test)[:, 1]
            X_test['ensemble_results'] = xgb_proba
            print(full_df.tail())
            print(X_test.tail())
            evaluate_df = pd.merge(full_df[['Tournament', 'Date', 'Round', 'Winner', 'Loser', 'WRank', 'LRank',
                                            'WPts', 'LPts', 'AvgW', 'AvgL', 'Winner_H2H_Wins', 'Loser_H2H_Wins',
                                            'h2h_matches', 'h2h_win_share_sm', 'AbsPtsDiff',  'Pts_Ratio',
                                            ]], X_test[['ensemble_results']], left_index=True, right_index=True)
            evaluate_df = evaluate_df[evaluate_df['h2h_matches'] > 0]
            evaluate_df["ensemble_profit"] = np.where(evaluate_df['ensemble_results'] > 0.5, evaluate_df["AvgW"] - 1, -1)
            # evaluate_df = evaluate_df['ensemble_results']
            print(evaluate_df)
            print(evaluate_df['ensemble_profit'].sum())


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



    def get_h2h_feature(self):
        def unordered_pair(a, b):
            return tuple(sorted([a, b]))

        def cumulative_wins_prior(df, key_col, winner_role_col):
            """
            df: sorted by Date
            key_col: 'ordered_key_w' or 'ordered_key_l'
            winner_role_col: boolean/0-1 indicator whether the first element of key won
            """
            # Build a frame with the ordered key and "did_first_win"
            tmp = df[[key_col]].copy()
            tmp['did_first_win'] = df[winner_role_col].astype(int)

            # Group by pair, build cumulative sum and then shift to represent "prior"
            tmp['cum_wins'] = tmp.groupby(key_col)['did_first_win'].cumsum()
            tmp['cum_wins_prior'] = tmp.groupby(key_col)['cum_wins'].shift(fill_value=0)
            tmp['cum_matches'] = tmp.groupby(key_col).cumcount()  # prior matches count

            return tmp[['cum_wins_prior', 'cum_matches']]


        for tour_name, temp_df in self.full_data.items():
            temp_df['pair_key'] = [unordered_pair(w, l) for w, l in zip(temp_df['Winner'], temp_df['Loser'])]
            temp_df['ordered_key_w'] = list(zip(temp_df['Winner'], temp_df['Loser']))  # winner perspective
            temp_df['ordered_key_l'] = list(zip(temp_df['Loser'], temp_df['Winner']))  # loser perspective

            # Sort by date to build leakage-free cumulative stats
            temp_df = temp_df.sort_values(['Date']).reset_index(drop=True)
            # print(temp_df.head())

            temp_df['did_first_win_w'] = 1

            # For loser perspective: ordered_key_l = (Loser, Winner)
            # The "first" player in ordered_key_l is the match loser, so did_first_win = 0 for every row.
            temp_df['did_first_win_l'] = 0

            # Compute prior cumulative wins for winner->loser
            w_stats = cumulative_wins_prior(temp_df, 'ordered_key_w', 'did_first_win_w')
            l_stats = cumulative_wins_prior(temp_df, 'ordered_key_l', 'did_first_win_l')

            # Attach to df
            temp_df[['h2h_wins_w_prior', 'h2h_matches_w_prior']] = w_stats.values
            temp_df[['h2h_wins_l_prior', 'h2h_matches_l_prior']] = l_stats.values

            # Sanity: for an unordered pair, prior total matches is the same from either side
            temp_df['h2h_total_prior'] = temp_df['Winner_H2H_Wins'] + temp_df['Loser_H2H_Wins']
            temp_df['is_first_meeting'] = (temp_df['h2h_total_prior'] == 0).astype(int)


            # Beta prior parameters (tunable). Start with a mild prior: alpha=beta=1 or 2.
            alpha_prior = 1.5
            beta_prior = 1.5

            # Winner's smoothed H2H win rate vs this opponent prior to the match
            # temp_df['h2h_win_pct_w_smooth'] = (
            #         (temp_df['h2h_wins_w_prior'] + alpha_prior) /
            #         (temp_df['h2h_total_prior'] + alpha_prior + beta_prior).replace(0, np.nan)
            # ).fillna(0.5)  # if no prior matches, prior mean = alpha/(alpha+beta) = 0.5
            #
            # # Loser's smoothed H2H win rate vs this opponent prior to the match
            # temp_df['h2h_win_pct_l_smooth'] = (
            #         (temp_df['h2h_wins_l_prior'] + alpha_prior) /
            #         (temp_df['h2h_total_prior'] + alpha_prior + beta_prior).replace(0, np.nan)
            # ).fillna(0.5)


            temp_df['h2h_win_pct_w_smooth'] = (
                    (temp_df['Winner_H2H_Wins'] + alpha_prior) /
                    (temp_df['h2h_total_prior'] + alpha_prior + beta_prior).replace(0, np.nan)
            ).fillna(0.5)  # if no prior matches, prior mean = alpha/(alpha+beta) = 0.5

            # Loser's smoothed H2H win rate vs this opponent prior to the match
            temp_df['h2h_win_pct_l_smooth'] = (
                    (temp_df['Loser_H2H_Wins'] + alpha_prior) /
                    (temp_df['h2h_total_prior'] + alpha_prior + beta_prior).replace(0, np.nan)
            ).fillna(0.5)

            # Optional: relative H2H tendency (centered)
            temp_df['h2h_w_relative_w'] = temp_df['h2h_win_pct_w_smooth'] - 0.5
            temp_df['h2h_w_relative_l'] = temp_df['h2h_win_pct_l_smooth'] - 0.5
            #
            #
            # print(temp_df[(temp_df['Winner_H2H_Wins'] > 0) & (temp_df['Loser_H2H_Wins'] > 0)][relevant_columns].tail(10))
            # print(temp_df[(temp_df['Winner_H2H_Wins'] == 0) & (temp_df['Loser_H2H_Wins'] == 0)][relevant_columns].tail(10))

            self.full_data[tour_name] = temp_df



    def h2h_feature_validation(self):
        for tour_name, temp_df in self.full_data.items():
            evaluate_df = temp_df.copy()
            # evaluate_df["expected_gain"] = np.where(evaluate_df['p_winner'] > 0.5, evaluate_df["AvgW"] - 1, -1)

            # NOTE: Check if the probability beats the Bet365 Odds and set Confidence level, the higher it is, the bigger the discrepancy between our prediction and betting odds
            disc_level = 0.2
            realistic_level = 5

            h2h_total = 1
            book_fav = 'B365'

            evaluate_df['p_h2h_feature_winner'] = evaluate_df['h2h_win_pct_w_smooth']
            evaluate_df['p_h2h_feature_loser'] = evaluate_df['h2h_win_pct_l_smooth']

            evaluate_df = evaluate_df[evaluate_df['is_first_meeting'] != 1]
            evaluate_df["winner_margin"] = evaluate_df["p_h2h_feature_winner"] - evaluate_df["implied_W"]
            evaluate_df["loser_margin"] = evaluate_df["p_h2h_feature_loser"] - evaluate_df["implied_L"]


            winning_df = evaluate_df[(evaluate_df["winner_margin"] > disc_level)
                                     & (evaluate_df[f"{book_fav}W"] < realistic_level)
                                     & (evaluate_df[f"h2h_total_prior"] >= h2h_total)
                                     ]

            losing_df = evaluate_df[(evaluate_df["loser_margin"] > disc_level)
                                    & (evaluate_df[f"{book_fav}L"] < realistic_level)
            & (evaluate_df[f"h2h_total_prior"] >= h2h_total)]

            winning_df['winning_money'] = winning_df[f'{book_fav}W'] - 1
            print(f'Tour Name: {tour_name}')
            print(f'Winning Money: {winning_df["winning_money"].sum()}')
            print(f'Losing Money: {len(losing_df)}')
            print(winning_df[relevant_columns].tail())
            print(losing_df[relevant_columns].tail())





    def run_analysis(self):
        self.h2h_get()

        self.get_rank_feature()
        self.rank_feature_validation()
        self.get_h2h_feature()
        self.h2h_feature_validation()

        # self.h2h_feature_test()
        # self.ranking_get()
        # self.ranking_stats()
        # self.model_evaluation()


        # self.ranking_construct()
        # self.h2h_feature_engineering()
        # self.ranking_stats_construct()
        # self.model_fitting(feature_cols=['Pts_Ratio'], target_variable='HRankWins', confidence_level=0.65)
        # self.elo_construct(elo_start_date=datetime.date(2024,12,20))
        # self.elo_strategy()





