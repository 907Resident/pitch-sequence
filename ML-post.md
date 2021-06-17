# Multi-Label Classification w/ Machine Learning - pitch-sequence

## Classifying Baseball Events
Top-line objective: *Using an assortment of pitching information, how well can we predict the specific outcome of an at-bat?*

This post is a part of a two-part series in which the MLB Pitch dataset is explored and used to learn ML classification techniques. This is the second part, which portrays the actial ML modeling that was conducted. See [pitch-sequence repo](https://github.com/907Resident/pitch-sequence) for more details.

## Preface
As an awardee of the [Kaggle BIPOC Program](https://www.kaggle.com/bipoc-grant-application), I had the opportunity to select a project that interested me and work on it directly with an established Machine Learning Engineer. Overall, I enjoyed my experience and would recommend others who satisfy the criteria for the program to apply in the future. I will say that it is tough to get all of what you want completed when you are also trying to finish a dissertation and start a new job. But I believe the following is a good synposis of how one can put ML to work through this program. 

## Introduction
For a brief recap, through this project, the goal for me was to learn more about how to improve multioutput classification models through the analysis of pitches thrown by MLB picthers. Please read [part one](https://github.com/907Resident/pitch-sequence/blob/main/EDA-post.md) for more details.

In this post, we will use the diffferent features of this dataset and then proceed to create a multi-label classification model. The model incorporates the data from the pitches thrown during the MLB season and then attempts to predict the outcome of the at-bat. This was an educational exercise, where I sought to learn how to iterate and improve a multi-label classification model through numerous attempts. The absolute predictive power of the model was a secondary objective and is not the focus of this post.

## The Data
This section is a duplication of what was written in the EDA post. However, it has been placed here for the reader's convenince. 

Within the posted dataset, there are three .csv files of importance: 1) pitches.csv, 2) atbats.csv, and 3) player_names.csv (note: in the second version of the dataset, Schale added data from the 2019 season in seperate .csv for the 1 and 2). In the pitches .csv, there are almost four dozen columns of data that range from `px`, the location along the axis of the pitch, to `on_1b` indicating that there is a baserunner on first base. However, the most relevant metrics in the pitching dataset describe the type of pitch thrown (`pitch_type`), number of strikes in the count (`s_count`), and number of balls in the count (`b_count`). Of course, when we proceed to modeling the outcome of the at-bat (`event`) will be the target variable. In the meantime, let's explore the variation among the pertinent metrics and features in the dataset.

## Preprocess the Data
More instructions on how to obtain the original dataset are shared in part one. Also, further discussion on how to import the data is shared there as well. We will proceed on the basis that the data has been succuessfully imported into the pyhton workspace as `df_main`.

Let's start by visualizing the the target label (`event`) through a countplot. 

```
# Target variable: Graph the frequency of outcomes
sns.set_theme(context="notebook", style="darkgrid", palette="gist_ncar_r")

# Events
plt.figure(figsize=(20,12))
sns.countplot(y="event", data=df_main, orient="h")
plt.xscale("linear")
plt.show()
```

## Mapped data: Lumping similar together
It is pretty clear that strikeout leads the way followed by ground, single, and walk. However, we can use our baseball knowledge to better make sense of which catrgories are similar. Since some events are relatively low, we will combine them together to increase the likelihood that our model can predict the "lumped" together `event`. 

Thus, combining the data above and knowledge of baseball, we get the following six outcomes for our target variable
- Non-HR Hit <- Single, Double, Triple
- HR < - Home Run
- Defensive Out <- Groundout, Flyout, Lineout, Pop Out, Forceout, Grounded Into DP, Double Play, Bunt Groundout, Fielders Choice, Fielders Choice Out, Bunt Lineout, Bunt Pop Out, Triple Play
- Strikeout <- Strikeout, Strikeout Double Play
- Walk <- Walk, Hit by Pitch, Intent Walk
- Sacrifice <- Sac Fly, Sac Bunt, Sac Fly Double Play, Sac Bunt Double Play
- Other <- Field Error, Catcher Interference, Batter Interference, Fan Interference

Sacrifice plays are an interesting subcategory because the rely on the presence and location of baserunners, the score of the game, and the prowress of the batter. Sacrifice events ($n = 5208$) such as:
- Sac Fly
- Sac Bunt
- Sac Fly Double Play
- Sac Bunt Double Play

will be scruitinzed carefully due to the fact these are not registered statistically as at bats. Furthermore, it will be intriguing to see how `pitch_type`, a charactristic of the pitcher, will relate to this play that is largely guided by factors external to the pitchers decision to throw a a specific pitch.

Similarly, plays that do not require the interaction between the pitcher and the batter are also dropped. For example, the [pickoff play](https://en.wikipedia.org/wiki/Pickoff) involves the pitcher throwing to a fellow defensive player to get an out. This does not require the interaction between the pitcher and the batter. Therefore the following inter-pitch events ($n = 801$) are also dropped: 
- Caught Stealing 2B
- Pickoff Caught Stealing 2B
- Pickoff 1B
- Caught Stealing Home
- Caught Stealing 3B
- Pickoff 2B
- Pickoff Caught Stealing Home
- Wild Pitch
- Pickoff 3B
- Pickoff Caught Stealing 3B
- Passed Ball
- Pickoff Error 1B
- Stolen Base 2B
- Runner Double Play
- Runner Out

```
# Create new dataframe "df_prepped"
df_prepped = df_main.reset_index(drop=True)

# List of non-at bat events
non_at_bat_subs_lst = \
["Caught Stealing 2B", "Pickoff Caught Stealing 2B", "Pickoff 1B", 
 "Caught Stealing Home", "Caught Stealing 3B", "Pickoff 2B", 
 "Pickoff Caught Stealing Home", "Wild Pitch", "Pickoff 3B", 
 "Pickoff Caught Stealing 3B", "Passed Ball", "Pickoff Error 1B", 
 "Stolen Base 2B", "Runner Double Play", "Runner Out"]

# Drop non-at bat events
mask = df_prepped.event.isin(non_at_bat_subs_lst)
df_prepped.event.iloc[mask] = np.nan
df_prepped = df_prepped[df_prepped["event"].notna()]

# Preview the loss of rows
df_prepped.shape
```

Roughly 901 rows (0.13 %) were dropped when the non-at bat scenarios were removed from the dataset. These seem a little low because Bill James [reports](https://www.billjamesonline.com/legally_stolen_bases/#:~:text=The%202019%20Philadelphia%20Phillies%20allowed%2066%20stolen%20bases%2C,is%20the%20starting%20point%20of%20our%20process%20here.) that the Philadelphia Phillies caught 50 runners stealing bases in 2019. Expanded to all 30 teams that would mean 1500 runners were caught stealing, which almost doubles the amount here. Finding caught stealing stats proves to be difficult.

```
# Organize the subevents into lists
## Non-Hit HR
non_hit_HR_subs_lst = ["Single", "Double", "Triple"]
## HR
HR_subs_lst = ["Home Run"]
## Defensive Out
defensive_out_subs_lst =\
 ["Groundout", "Flyout", "Lineout", "Pop Out",  "Forceout", "Grounded Into DP", 
  "Double Play","Bunt Groundout", "Fielders Choice", "Fielders Choice Out", 
  "Bunt Lineout", "Bunt Pop Out", "Triple Play"]
## Strikeout
strikeout_subs_lst = ["Strikeout", "Strikeout Double Play"]
## Walk
walk_subs_lst = ["Walk", "Hit By Pitch", "Intent Walk"]
## Sacrifice
sac_subs_lst = ["Sac Fly", "Sac Bunt", 
                "Sac Fly Double Play", "Sac Bunt Double Play"]
## Other
other_subs_lst = ["Field Error", "Catcher Interference", "Batter Interference",
                  "Fan Interference"]
# Create empty series for updated events
event_new = pd.Series(index=range(len(df_main)))

# Map subcategories of event
relev_events = ["Non-HR Hit", "HR", "Def Out", "Strikeout", "Walk", "Sacrifice",
                "Other"]

# Use boolean masks to map the data accordingly
## Non-Hit HR
mask = df_main.event.isin(non_hit_HR_subs_lst)
event_new.iloc[mask] = relev_events[0]
## HR
mask = df_main.event.isin(HR_subs_lst)
event_new.iloc[mask] = relev_events[1]
## Defensive Out
mask = df_main.event.isin(defensive_out_subs_lst)
event_new.iloc[mask] = relev_events[2]
## Strikeout
mask = df_main.event.isin(strikeout_subs_lst)
event_new.iloc[mask] = relev_events[3]
## Walk
mask = df_main.event.isin(walk_subs_lst)
event_new.iloc[mask] = relev_events[4]
## Sacrifice
mask = df_main.event.isin(sac_subs_lst)
event_new.iloc[mask] = relev_events[5]
## Other
mask = df_main.event.isin(other_subs_lst)
event_new.iloc[mask] = relev_events[-1]

# Add event_new to dataframe 
df_prepped.insert(0, "event_new", event_new)
```

From a previous iteration of the ML model, we know that `code` is an unfair feature to use because it effectively records what happens after the pitch is thrown. Since the objective is predict what pitches and scenarios lead to the various events, `code` is dropped from the dataframe.

```
# Drop "code"
df_prepped = df_prepped.drop(axis=1, columns=["code"])

# Also, drop the un-lumped "event", "first_name", "last_name", "pitcher_id", 
# and "ab_id" columns. "event_new" will be the new target variable and "ab_id"
# is not useful 
df_prepped = df_prepped.drop(axis=1, columns=["event", "pitcher_id", "ab_id", 
                                              "first_name", "last_name"])

# Visualize lumped events
fig = plt.figure(figsize=(12,8))
sns.countplot(y="event_new", data=df_prepped, orient="h")
plt.xscale("linear")
plt.show()
fig.savefig("figures/lumped_events_2019.png", dpi=300)                                              
```

## Feature Engineering
In this exercise there was not too much in the way of feature engineering. However, one feature that was generated is the difference in the score between the pitcher and batter's respective teams (`score_delta`). 

```
# Create a feature score_delta that relates the score of the game relative to pitcher
df_prepped["score_delta"] = df_prepped.p_score - df_prepped.b_score

# Export to csv
df_prepped.to_csv("df_prepped.csv", sep=",", index=False)
```

## Encode categorical variables
There are several categorical variables in this dataset including the target variable, "event." For the ML algorithms to prcesss them appropriately, they need to be encoded as numeric values. Further guidance on incorporating categorical variables into ML models was gleaned from [Frank and Hall (2001)](https://www.cs.waikato.ac.nz/~eibe/pubs/ordinal_tech_report.pdf) and [Chen et al.](https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf).

### Target Variable: `event_new`
To use `sklearn` we will need to apply an encoder to ensure that the target is recorded as numerical information instead of text

The `sklearn` [`label_encoder()`](https://sklearn.org/modules/generated/sklearn.preprocessing.LabelEncoder.html#:~:text=sklearn.preprocessing%20.LabelEncoder%20%C2%B6%20%20%20fit%20%28y%29%20,of%20this%20estimator.%20%201%20more%20rows%20) works well for this purpose. 

```
from sklearn.preprocessing import LabelEncoder
# Instantiate the encoder
target_encoder_le = LabelEncoder()

# Fit and transform the encoder to the target variable
## Separate target variable from explanatory features
y = df_prepped.iloc[:,0].values
## Apply encoder
y = target_encoder_le.fit_transform(y.astype(str))
```

### Features: All variables that are not the target

Just like the target variable, the categorical variables from the explanatory data (i.e. features) will need to transformed with encoding.

```
# Instantiate the encoder
explanatory_encoder_le = LabelEncoder()

# Fit and transform the encoder to the explanatory variable
## Separate explanatory variables from the target variable
X = data_subsample.iloc[:, 1:].values

## Apply encoder
### pitch_type
X[:,6] = explanatory_encoder_le.fit_transform(X[:,6].astype(str))
### b_count
X[:,8] = explanatory_encoder_le.fit_transform(X[:,8].astype(str))
### s_count
X[:,9] = explanatory_encoder_le.fit_transform(X[:,9].astype(str))
### p_throws
X[:,14] = explanatory_encoder_le.fit_transform(X[:,14].astype(str))
```

## Execute ML Algorithms
As mentioned earlier, multiple iterations of the ML algorithms were used to gauge the importance of the parameters used in this project. To separate these model runs the following numerical system was employed: IterationNumber.IterationModification.IterationSubModification. The largest changes to the approach are noted by the `IterationNumber`, followed by the `IterationModification`, and the most minor changes indicated by `IterationSubModification`. Not all attempts are presented in those. The attempts not discussed in the post can be found in detail in the notebook. 

- iteration 1.0.0: H2O only
- iteration 2.0.0: RF Classifier with max depth at 2 and seed at 42 (`code`) dropped as discussed with Josh
- iteration 2.1.0: grid search across parameters for RF
- iteration 2.2.0: One-vs-Many SVM classifier