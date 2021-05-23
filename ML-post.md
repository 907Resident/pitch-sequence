# Multi-Label Classification w/ Machine Learning - pitch-sequence

## Classifying Baseball Events
Top-line objective: *Using an assortment of pitching information, how well can we predict the specific outcome of an at-bat?*

This post is a part of a two-part series in which the MLB Pitch dataset is explored and used to learn ML classification techniques. See [pitch-sequence repo](https://github.com/907Resident/pitch-sequence) for more details.

## Preface
As an awardee of the [Kaggle BIPOC Program](https://www.kaggle.com/bipoc-grant-application), I had the opportunity to select a project that interested me and work on it directly with an established Machine Learning Engineer. Overall, I enjoyed my experience and would recommend others who satisfy the criteria for the program to apply in the future. I will say that it is tough to get all of what you want completed when you are also trying to finish a dissertation and start a new job. But I believe the following is a good synposis of how one can put ML to work through this program. 

## Introduction
About 730k pitches are thrown in Major League Baseball (MLB) every season. Thanks to [Paul Schale](https://www.kaggle.com/pschale), a [dataset](https://www.kaggle.com/pschale/mlb-pitch-data-20152018?select=2019_pitches.csv) of these picthes from the 2015-2019 seasons has been produced for all to access on Kaggle. On Kaggle there have been numerous notebooks made to analyze and model the data. One [user](https://www.kaggle.com/markben/mlb-considering-handedness-for-pitchers-batters) examined how the handedness of a pitcher could affect the outcome of an at-bat (hit or an out). Whereas another another [user](https://www.kaggle.com/pschale/bayesian-inference-for-strike-zone-analysis) used bayesian analysis to uncover a slight (~1 inch) change in the size of the strike zone after an ejection occurs.

In this post, we will the diffferent features of this dataset and then proceed to create a multi-label classification model. The model incorporates the data from the pitches thrown during the MLB season and then attempts to predict the outcome of the at-bat. This was an educational exercise, where I sought to learn how to iterate and improve a multi-label classification model through numerous attempts. The absolute predictive power of the model was a secondary objective and is not the focus of this post.

## The Data
Within the posted dataset, there are three .csv files of importance: 1) pitches.csv, 2) atbats.csv, and 3) player_names.csv (note: in the second version of the dataset, Schale added data from the 2019 season in seperate .csv for the 1 and 2). In the pitches .csv, there are almost four dozen columns of data that range from `px`, the location along the axis of the pitch, to `on_1b` indicating that there is a baserunner on first base. However, the most relevant metrics in the pitching dataset describe the type of pitch thrown (`pitch_type`), number of strikes in the count (`s_count`), and number of balls in the count (`b_count`). Of course, when we proceed to modeling the outcome of the at-bat (`event`) will be the target variable. In the meantime, let's explore the variation among the pertinent metrics and features in the dataset.

## Importing the Data into Python
In this section, import the data into dataframes, assess missing data, and clean data for exploratory analysis to take place in the next session. 
### Connect to Google Drive
For this project, I used [Google Colab](https://colab.research.google.com/) because it allows me to train models on Google's hardware freeing up my personal machine for other uses. Furthermore, it allows the sharing of notebooks with no installing or downloading of external software; the recipient of your notebook need only a browser and internt connection to view and run your code. Of course read through the documentation and [FAQs](https://research.google.com/colaboratory/faq.html) if you have concerns about using this platform for your work. 

Link to Google Drive so that zip files procured from Kaggle API sits in Google Drive instead of local machine to save space and for flexibility.  More details on how to set this up can be found on this Medium [post](https://medium.com/analytics-vidhya/how-to-fetch-kaggle-datasets-into-google-colab-ea682569851a) by Mrinali Gupta.
```
# Connect to Google Drive
from google.colab import drive
drive.mount('/content/gdrive')
```

### Access data from Kaggle API
Kaggle has organized an API so that data, competition submissions, and other related materials can be accessed from a command interface. In order to acquire the data for this project, the Kaggle API was utilized. Read more [here](https://github.com/Kaggle/kaggle-api) about the API on GitHub. *Important:* Create an environmental variable for the kaggle config file for Kaggle API get credentials. Follow the post mentioned above for specific instructions on how to access the necessary .json file.

One can also download data manually from the Kaggle website; however, building in this process to your code will allow you to pick up any new versions of the data and keep a better record of where and how the data was accessed as well as its origin.

```
import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Colab Notebooks/pitch-sequence"
```

Point current directory towards the appropriate directory in Google Drive
```
%cd /content/gdrive/My Drive/Colab Notebooks/pitch-sequence
```

Install the Kaggle API into your workspace
```
# Install Kaggle API
!pip install -q --upgrade kaggle
```

Download the data from the API
```
# Download the pitches and at bats data set from 2019
!kaggle datasets download pschale/mlb-pitch-data-20152018 -f 2019_pitches.csv
!kaggle datasets download pschale/mlb-pitch-data-20152018 -f 2019_atbats.csv
!kaggle datasets download pschale/mlb-pitch-data-20152018 -f player_names.csv

# Download the pitches and at bats data set from 2015-2018
!kaggle datasets download pschale/mlb-pitch-data-20152018 -f pitches.csv
!kaggle datasets download pschale/mlb-pitch-data-20152018 -f atbats.csv
```

### Import the data

```
# Import data science packages
import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
```

Begin with the pitches
```
# Import data from 2019 .csv
df_pitches_19 = pd.read_csv("/content/gdrive/My Drive/Colab Notebooks/pitch-sequence/2019_pitches.csv.zip")
# Take a look at the dimensions of the freshly imported dataframe.
df_pitches_19.shape
```

At-bats
```
# Import data from competition .csv
df_atbats_19 = pd.read_csv("/content/gdrive/My Drive/Colab Notebooks/pitch-sequence/2019_atbats.csv.zip")
# Take a look at the dimensions of the freshly imported dataframe.
df_atbats_19.shape
```

Names of the players
```
df_names = pd.read_csv("/content/gdrive/My Drive/Colab Notebooks/pitch-sequence/player_names.csv")
# Take a look at the dimensions of the freshly imported dataframe.
df_names.shape
```

Join the pitch and at bat data  to get the pitcher ID into the former dataframe. This will allows us to know who is throwing the pitches. Drop all columns from at-bat dataframe (because we mostly interested in the pitching aspect) except pitcher ID, pitcher throws (p_throws), and event.
```
# Join the pitches and at bat dataframes
df_merged = df_pitches_19.merge(df_atbats_19, how="left", on="ab_id")

# Drop unncessary at bat columns
df_merged.drop(["o", "stand", "batter_id", "g_id", "top"], 
               inplace = True, axis = 1)

# Preview dataframe
df_merged.head()
```

Drop some of the superfluous columns from the pitches dataset. Definitions of the columns can be found [here](https://fastballs.wordpress.com/2007/08/02/glossary-of-the-gameday-pitch-fields/).
```
# Drop extraneous picthing columns
df_merged.drop(["break_angle", "break_length", "break_y", "ax", 
                "ay", "az", "vx0", "vy0", "vz0", "x", 
                "x0", "y", "y0", "z0", "pfx_x", "pfx_z", "zone", "type",
                "event_num", "on_1b", "on_2b", "on_3b"], 
        inplace = True, axis = 1)

# Preview dataframe
df_merged.head()
```

```
# Get the new shape of the main dataframe
df_merged.shape
```

Merge the player names to the the dataframe
```
# Merge main df to the player names with pitcher ID
df = df_merged.merge(df_names, how="left", left_on="pitcher_id", right_on="id")

# Drop "id" column from player names dataframe
df.drop(["id"],
        inplace = True, axis = 1)

# Preview main dataframe
df.head()
```

Reorder the dataframe so that the pitcher IDs and their names at at far left of the dataframe. Currently, there is not a simply way to move the columns;
therefore: 

*   move the necessary columns to the left with the `insert()` function with misspelled titles
*   delete the original column
*   rename the purposefully misspelled columns to the correct spelling
```
# Move pitcher ID to the front
df.insert(0, "pitche_id", df.pitcher_id)

# Move First Name next to pitcher ID
df.insert(1, "firs_name", df.first_name)

# Move Last Name next to First Name
df.insert(2, "las_name", df.last_name)

# Drop Duplicates
df.drop(labels=["pitcher_id", "first_name", "last_name"], 
                   axis=1, inplace=True)

# Rename misspled column names
df.rename(columns = {'pitche_id':'pitcher_id', 
                     'firs_name':'first_name',
                     'las_name':'last_name'}, 
          inplace = True)

# Fill missing data with "NA"
#df.fillna(value="NA", inplace=True)

# Preview dataframe
df.head()
```

## Exploratory Data Analysis
With a clean dataset of pitches during the 2019 season, it is time to dive in and analyze. 

### Visualize important metrics
Start by getting a feel of the important metrics in the dataset. Some of these important metrics are:


*   `start_speed`
*   `end_speed`
*   `event`
*   `pitch_type`
*   `b_count`
*   `s_count`
*   `last_name`

Best place to start is understanding which is the most common pitch type
```
# Set graph style
sns.set_style("darkgrid")

# Graph pitch types
## Absolute Number of Pitches
plt.figure(figsize=(4,4))
df.pitch_type.value_counts().plot(kind="bar", edgecolor="k")
plt.yscale("log")
plt.title("Number of Pitches Thrown by Type")
plt.ylabel("Num. Pitches Thrown")
plt.show()

## Density plot of pitch speed by pitch type
### Start Velocity
plt.figure()
sns.kdeplot(x=df["start_speed"], hue=df["pitch_type"])
plt.title("Initial Speed of Pitch")
plt.xlabel("Initial Speed (mph)")
plt.show()
### End Velocity
plt.figure()
sns.kdeplot(x=df["end_speed"], hue=df["pitch_type"])
plt.title("Final Speed of Pitch")
plt.xlabel("Final Speed (mph)")
plt.show()
```

Examine what pitches are thrown in different situations. A mosaic plot is great for plotting two catgeorical variables.  Get the `mosaic()` function from the `statsmodels.graphics` module

For instance, when there is 0, 1, or 2 outs, what pitch is thrown?
```
# Mosaic plot for pitch type and outs in the count
## Import mosaic function
from statsmodels.graphics.mosaicplot import mosaic

# Create marginal table
plt.rcParams["figure.figsize"]=(16, 8)
mosaic(df, ["pitch_type", "outs"],
       gap = 0.008,
       title="Pitch Type and Outs")
plt.ylabel("Outs")
plt.show()
```

The mosaic plot shows that the four-seam fastball (FF) is dominant pitch but also shows an approximate equal distribution of FF pitches in all three out scenarios. 

Furthermore, the mosaic plot shows that the slider (SL) and knukle curve (KC) are thrown more often with two outs than with one or zero. Also, the chart also demonstrates that the SL is thrown far more often than the KC based on the width of the bars. Therefore, we now have some information about how the game may affect pitch type,

Using a series of pie plots, we can examine the pitch selected and how it corresponds to the number of strikes and balls in the count. In these pie charts, the pitch type label is given as well as the number of strikes and balls in the count.

```
ptype_labs = ["FF", "SL", "CH", "CU", "FC", "FT", "SI", "FS", "KC", "FO", "KM",
              "EP"]
df_count_ptype = df_bls_strx.groupby(["b_count", "s_count"]).pitch_type.value_counts()

cmap = "Set3"
fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(12,8))
# 0,0 count
df_count_ptype[:12].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                         ax=ax[0,0])
ax[0,0].set_ylabel('')
# 0,1 count
df_count_ptype[13:24].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                           ax=ax[0,1])
ax[0,1].set_ylabel('')
# 0,2 count
df_count_ptype[25:35].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                           ax=ax[0,2])
ax[0,2].set_ylabel('')
# 1,0 count
df_count_ptype[36:47].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                           ax=ax[1,0])
ax[1,0].set_ylabel('')
# 1,1 count
df_count_ptype[49:59].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                           ax=ax[1,1])
ax[1,1].set_ylabel('')
# 1,2 count
df_count_ptype[61:70].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                           ax=ax[1,2])
ax[1,2].set_ylabel('')
# 2,0 count
df_count_ptype[71:82].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                           ax=ax[2,0])
ax[2,0].set_ylabel('')
# 2,1 count
df_count_ptype[82:94].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                           ax=ax[2,1])
ax[2,1].set_ylabel('')
# 2,2 count
df_count_ptype[94:103].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                            ax=ax[2,2])
ax[2,2].set_ylabel('')
# 3,0 count
df_count_ptype[103:114].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                             ax=ax[3,0])
ax[3,0].set_ylabel('')
# 3,1 count
df_count_ptype[114:125].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                             ax=ax[3,1])
ax[3,1].set_ylabel('')
# 3,2 count
df_count_ptype[125:134].plot(kind="pie", autopct='%1.0f%%', cmap=cmap,
                             ax=ax[3,2])
ax[3,2].set_ylabel('')

f = plt.gcf()
f.set_size_inches(12,14)
plt.tight_layout()
plt.show()
```
### Two-strike counts
In situations where there are two strikes in the count, the diversity of pitches expands relative to earlier in the count.
```
df_k_2S = df[(df["s_count"] == 2)]
df_k_2S.drop_duplicates(subset=["ab_id"], keep="last", inplace=True)
df_k_2S.reset_index(inplace=True, drop=True)

df_k_2S_cnt = df_k_2S.pitch_type.value_counts()
df_k_2S_cnt.plot(kind="pie", autopct='%1.0f%%', cmap=cmap)
plt.ylabel('')
plt.show()
```
This pie chart shows the probability of pitch type selected when the result is a strikeout and there are two strikes in the count. To be clear, this is the pitch that was selected that resulted in the strikeout. 
```
(df_k_2S["event"]== "Strikeout").mean()
```
This quick calculation shows that about half of the time the data shows that pitcher is able to strikeout the batter when there are two strikes in the count.

We can explore this further by analyzing the `event` or outcome from each of the 2-strike at-bats. 
```
df_k_2S["is_SO"] = (df_k_2S["event"]== "Strikeout")
df_k_2S.groupby(["pitch_type"])["is_SO"].mean().sort_values(ascending=False)
```

This simple analysis demonstrates the probability depending on pitch type selection for 2-strike counts that end in a strikeout. Though all of the pitch types represented are not thrown in equal amounts (see EDA), the knucklecurve (KC) results in a strikeout almost 60% of the time its is thrown in a 2-strike count scenario. More common pitches, such as the slider (SL), curveball (CU), four-seam fastball (FF), change up (CH), and splitter (FS) result in strikeouts in the same count situation between 51 and 55% of the time. Surprisingly, sinkers (SI) only result in strikeouts 43% of the time suggesting that batters take (do not swing) at that pitch.
```
# Crosstab analysis
df_ptype_xtab_event = \
pd.crosstab(df_k_2S["pitch_type"], df_k_2S["event"],
            normalize="index").\
drop(columns=["Batter Interference", "Catcher Interference", 
              "Caught Stealing 2B", "Caught Stealing Home", 
              "Pickoff Caught Stealing Home", "Runner Out",
              "Sac Fly Double Play"])
## Ensure column names are unattached to "event"
df_ptype_xtab_event.columns = df_ptype_xtab_event.columns.to_list()

# Simple heatmap
sns.heatmap(df_ptype_xtab_event, cmap="rainbow", linewidths=0.5, linecolor="w",
            annot=True)
```

This heatmap highlights the terminal scenarios that occur after the pitch is thrown in the two strike count. It is relatively clear, as mentioned earlier, that the result ends in a strike out about half the time for all pitches. The only exception is the knuckleball (KN), which appears to have only been thrown a few times (possibly only once) and it hit a batter. Again, the curveball (CU) and slider (SL) are the pitches that lead to a strikeout in this 2-strike count the most often.

The clustermap examines how the pitch type and event types are related to one another. The clustermap clearly sets apart the knuckleball (KN) from the rest of the pitches. Also the pitchout (FO) is also seperated from the rest of the pitches (relevant pitches) as well. It appears, from the event side of things, grounding out, getting a single or a walk are all loosely related to the most relevant pitch types. Grounding out is slighly seperate from getting a single or a walk.

As the heatmap indicated, there is not a huge difference between the relevant pitches in terms of strikeout probability if you allow for a 5% difference between piches. However, the clustermap reveals that there is a correlative relationship between sinkers (SI) and cutters (FC) as well as curveballs (CU) and splitters (FS). 