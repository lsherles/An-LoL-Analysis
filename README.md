# Is it Possible to Predict How a Top-Tier Competitive League of Legends Match Will Go?

By Lucas Sherles (lsherles@ucsd.edu)

## Introduction

In competitive League of Legends, the draft is make or break for a team's success. This is where each team selects and bans 5 champions, going in a snake order. The goal is to pick champions that are strong and synergize well with each other, while also banning champions that are strong against your own.

This project focuses on the the tier 1 leagues (LCK, LEC, LCS, LPL), which are the highest level of League of Legends anywhere around the world, and explores how the draft phase affects the outcome of each game. 

In the first few parts of this project, we will look at at the most picked and banned champions, and how they affect the win rate of the teams that pick or ban them. Later on, we will look at how the champions picked in the draft can predict which team obtains the first dragon in the game.

---

## Cleaning and EDA

The first step of the cleaning part of this process was to query the dataset for only the top domestic leagues, not including international events. In the data, each entry is assigned to a specific player. Since we aren't interested in player-specific statistics, we can group the data by _gameid_ and _teamname_. This gives us the team statistics for each game, and includes entries for both teams. Specifically, let's figure out which champions were picked or banned the most times. 

Since there are 5 ban slots, each of which have their own column, we need to melt these columns into one column. We can do this by using _groupby_, and since we aren't interested in player-specific statistics, we can group the data by _gameid_ and _teamname_ and keep the relevant columns. Then, we can get the value counts to see which champions were banned the most, and when we combine this with the value counts of the 'champion' column in the original dataset, we can plot the results.

```py
print(pickban_df.head(5).to_markdown(index=False))
```

| champion    |     count    |
|:------------|-------------:|
| Ashe        |       1171.0 |
| K'Sante     |       1100.0 |
| Kalista     |       1096.0 |
| Rumble      |       1094.0 |
| Nautilus    |        967.0 |


<iframe
  src="assets/valcounts.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
As we can see, the champion that is picked or banned the most is Ashe. Let's explore Ashe's effect on how teams perform by seeing if teams experience more or less success when they pick or ban Ashe.

To explore Ashe's effect on how teams perform, we start by filtering the original dataframe for when '_champion_' is equal to Ashe, and then simply find the mean of the '_result_' column. To find the win rate for teams who banned, we just need to filter the previous dataframe that we melted to contain games where Ashe was banned, and then take the mean of '_result_'. We can then combine these dataframes to create one larger dataframed, '_gameid_' and '_teamname_', that contains a new column, '_status_', dictating whether Ashe was picked or banned by the specific team in the game. Games where Ashe was picked will, as a result, have a value of NaN under the '_ban slot_' column.  

```py
print(ashe_df.head(5).to_markdown(index=(False)))
```

|gameid	            |teamname	            |league	|result	|champion	|status	|ban_slot|
|:------------------|-----------------------|-------|-------|-----------|-------|-------:|
|10665-10665_game_1	|Bilibili Gaming	    |LPL	|1	    |Ashe	    |Picked	|NaN     |
|10665-10665_game_2	|Bilibili Gaming	    |LPL	|0	    |Ashe	    |Banned	|ban2    |
|10665-10665_game_3	|Top Esports	        |LPL	|0	    |Ashe	    |Banned	|ban2    |
|10666-10666_game_1	|Royal Never Give Up	|LPL	|0	    |Ashe	    |Banned	|ban3    |
|10666-10666_game_2	|Royal Never Give Up	|LPL	|1	    |Ashe	    |Banned	|ban1    |


<iframe
  src="assets/ashe_wr_bc.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
Interestingly enough, teams that ban Ashe not only lose more often than they win, but they also lose staggeringly more than teams who pick Ashe, who do very well.

---

## Assessment of Missingness

To prepare for the model, we'll explore the missingness of the _firstdragon_ column. For this, we will first look at the whole original dataset (containing all leagues, not just the tier 1 leagues), and then look at the tier 1 leagues separately. The values in the  _firstdragon_ column are missing for all individual player entries and is not missing for some entries pertaining to teams, so we'll filter out entries in the dataset that are related to individual players. This could mean that the data is NMAR and to figure out how the missingness of _firstdragon_ works, we can see if there is any relation to the _league_ column, as some leagues may not track which team gets the first dragon and other tedious data like so. 

<iframe
  src="assets/all_leagues_missingness_tvd.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

As we see in the graph, the total variation distance (TVD) of missingness in our permutation tests is very small, but the TVD in our observed data is nearly at 1, while the TVD in our permutation tests is near zero for all of the permutations. With a p-value of 0, the _firstdragon_ column's missingness appears to be dependent on the _league_ column. This would suggest that some leagues simply don't track this level of detailed information in their matches, and that the data is MAR.

<iframe
  src="assets/tier1_leagues_missingness_tvd.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Looking at just the tier 1 leagues, it is clear that there is one league out of the 4 that doesn't track this data. This league is the LPL, and when it comes to creating the predictive model, we will have to drop LPL data from the dataset when we try to predict _firstdragon_. With a p-value of 0, we can conclude that the missingness of the _firstdragon_ column is dependent on the _league_ column, making it MAR.

We might also consider that side selction has to do with the missingness of the _firstdragon_ column, so let's run a permutation test and explore the variance between the _side_ column and the missingness of _firstdragon_.

<iframe
  src="assets/side_missingness_var.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

Based on the graph and a p-value of 1.0, there is no evidence that _side_ has to do with the missingness of _firstdragon_.

---

## Hypothesis Testing

We will use hypothesis testing to see if the difference in win rates between teams who ban Ashe versus teams who pick Ashe is significant. Based on the previous graph comparing win rates for when Ashe is picked or banned, we can outline our test is outlined as follows:
- Null: The win rate of teams that pick Ashe is the same as the win rate of teams that ban Ashe.
- Alternative: The win rate of teams that pick Ashe is higher than the win rate of teams that ban Ashe.
- Significance Level: 0.05 or 5%

To test whether the difference is significant, we can use z-score to measure the standardized difference between the two. If our z-score is large, then we find that picking Ashe leads to a win a signficantly higher amount than banning Ashe. After calculating the total win rate and standard error, we can take the differences in individual win rates and divide by the SE to find the z-score. We can further use _norm.cdf_ to calculate the p-value as well.

```py
# Compute Z-score
z_score = (ashe_pick_wr - ashe_ban_wr) / SE

# Compute one-tailed p-value (right-tailed test)
p_value = 1 - norm.cdf(z_score)

print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")
Z-score: 1.9584
P-value: 0.0251
```


Based on these values and our significance level of 0.05, we recommend that the null hypothesis be rejected in favor of the alternative. This would suggest that picking Ashe gives teams a significantly better chance of winning than banning Ashe does.

---

## Framing a Prediction Problem

Having previously explored the effect of champions on win rates, it's time to explore a more in-depth model. Given that there are over 160 champions in League of Legends, some champions "spike" or are stronger at certain points in the game. For example, Kayle is considered to be a very strong champion in later stages of the game, as she gains new forms that are stronger as she reaches higher experience levels. Thus, it would not be a surprise to see games with Kayle being longer games, or teams who pick Kayle getting fewer void grubs. In this model, we will look at how the champions selected by a team in a draft affects whether or not the obtain the first dragon in the game. 

This will be a binary classification problem where we try to predict the _firstdragon_ column based on the champions picked, assigning a binary target variable that is 1 if we predict the team to get the first dragon, and 0 otherwise. To initialize our model, we will use all of the "pick" columns (labeled _pick1_, _pick2_, etc., up to 5). Since the draft happens prior to the start of the game, we can be sure that we would have all of the information on the "pick" columns when making our prediction about whether a team will get the first dragon. Since the _firstdragon_ column is perfectly balanced (half of the entries are 1s and the other half are 0s), we will use accuracy as our metric over f1-score. To do this, we will use one hot encoding for the champions that are picked, compounded with sklearn's _Pipeline_ and _RandomForest_.

---
## Baseline Model
As stated previously, we used one hot encoding for the champions picked. All of the pick columns are categorical, having names of champions as the values in these columns. We can use _ColumnTransformer_ to achieve this, and then define the pipeline using the previous one hot encoding as well as a _RandomForest_ in order to bootstrap and analyze the complex patterns that exist in the many combinations of potential champions. Then, we finally train the model, predict it using the test set, and check our accuracy which is as follows:

```py
# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
Model Accuracy: 0.5541
```

In order for the model to be better than if we were to just predict a 1 for every entry of _firstdragon_, we would need our accuracy to be better than 0.5 or 50%, since _firstdragon_ is perfectly balanced between 0s and 1s. We find that the model is better than this case, coming in at an accuracy of 55.41%. This would suggest that our model is good, but not great. The champions picked definitely help us to make predictions about whether a team will get the first dragon in a game, but our model is only slightly better than a base case. Let's see if we can improve upon that in our final model.

---

## Final Model
We add _side_ to our model because there is evidence that red side gets the first dragon of the game more often. This can be for multiple reasons, such as the team playing on red side having an easier entrance into the dragon pit on the map. Teams that are red side also end the draft with the last pick, which can be used to counter the enemy team's draft. They also can pick both champions going into their bottom lane as their first two picks, while the blue side can only pick one champion initially. This can give red side a stronger bot lane pairing going into the game, and since the dragon pit is on the bottom side of the map, teams with strong bot lanes are usually favored to take the first dragon. This is corroborated by the champion Varus having the highest importance in the model when he is in pick slot 1 (seen in the graph below), as he is a strong bot lane champion in the early game.

Furthermore, we also added _teamname_ into the model. This is an important variable to add because some teams might prioritize obtaining the first dragon of the game more than others. Some teams may have players in the bottom lane who like to play champions that are strong in the early game, which would lead to them obtaining the first dragon of the game more often than not, and vice versa. Let's take a look at the most predictive features of our model.


<iframe
  src="assets/finalmodel.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
---

Adding _side_ to our model certainly helped boost the accuracy of the model, as both values for the column are at the top of the most important features. There are also two teams that appear in the top 20 most important features, Fnatic and Team BDS, which would suggest that these teams likely have more polarizing numbers in terms of how often they are getting the first dragon of the game. Apart from those features, the next most important is "_onehot_pick1_varus_", which is when Varus is picked in the first slot. This would suggest to us that Varus is a storng champion in the early game, allowing for his team to be stronger and have more control when the first dragon spawns at 5:00 into the game. Interestingly as well, when Varus is picked first by red side, the first dragon rate shoots up to a staggering 57.53%. 

## Fairness Analysis

All that is left to do is to embed the confusion matrices, maybe create a graph for the permutation test here, embed that, and explain the findings, then am done.
---