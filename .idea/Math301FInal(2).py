import pandas as pd
from io import StringIO
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# Load the CSV data for team stats
data = """
Player,PF,Yds,Ply,Y/P,TO,FL,1stD,Cmp,Att,Yds,TD,Int,NY/A,1stD,Att,Yds,TD,Y/A,1stD,Pen,Yds,1stPy,#Dr,Sc%,TO%,Start,Time,Plays,Yds,Pts
Team Stats,375,5647,1030,5.5,24,8,329,413,593,4020,21,16,6.2,211,383,1627,17,4.2,88,119,987,30,182,35.2,9.9,Own 27.8,2:42,5.78,30.4,1.89
Opp. Stats,368,5656,1065,5.3,18,5,329,371,569,3603,26,13,5.9,193,451,2053,11,4.6,107,121,958,29,194,34.0,9.3,Own 29.7,2:45,5.7,29.3,1.78
Lg Rank Offense,18,14,,,24,14,16,,7,8,20,25,15,,29,28,11,17,,,,,,22,20,31,24,24,16,21
Lg Rank Defense,11,14,,,16,27,15,,19,11,17,12,6,,14,16,4,23,,,,,,26,19,14,13,4,6,5
"""

# Create a DataFrame from the CSV string
df = pd.read_csv(StringIO(data))

# Print the first few rows of the DataFrame to check the content
print(df.head())

# Data cleaning: Renaming the columns and handling any irregularities
df.columns = ['Category', 'PF', 'Yards', 'Plays', 'YardsPerPlay', 'TO', 'FL', '1stD',
              'PassingCmp', 'PassingAtt', 'PassingYds', 'PassingTD', 'PassingInt', 'PassingNYA', 'Passing1stD',
              'RushingAtt', 'RushingYds', 'RushingTD', 'RushingYA', 'Rushing1stD', 'PenaltiesPen',
              'PenaltiesYds', 'Penalties1stPy', 'AvgDriveDr', 'AvgDriveSc%', 'AvgDriveTO%',
              'AvgDriveStart', 'AvgDriveTime', 'AvgDrivePlays', 'AvgDriveYds', 'AvgDrivePts']

# Convert necessary columns to numeric values
df['PF'] = pd.to_numeric(df['PF'], errors='coerce')
df['Yards'] = pd.to_numeric(df['Yards'], errors='coerce')
df['TO'] = pd.to_numeric(df['TO'], errors='coerce')
df['Plays'] = pd.to_numeric(df['Plays'], errors='coerce')

# Check for missing or NaN values
print("Missing data summary:")
print(df.isna().sum())

# Clean up NaN or missing data (if applicable)
if df.empty:
    print("The DataFrame is empty after cleaning. Exiting the process.")
else:
    # Fill NaN values with 0 (you can also fill with the mean/median if appropriate)
    df.fillna(0, inplace=True)

    # Now apply scaling
    scaler = MinMaxScaler()

    # Check if the columns exist and are non-empty
    if all(col in df.columns for col in ['Yards', 'PF', 'RushingYds', 'PassingYds']):
        df[['Yards', 'PF', 'RushingYds', 'PassingYds']] = scaler.fit_transform(df[['Yards', 'PF', 'RushingYds', 'PassingYds']])
    else:
        print("One or more columns needed for scaling are missing.")

    # Print the DataFrame after scaling
    print("\nDataFrame after scaling:")
    print(df[['Yards', 'PF', 'RushingYds', 'PassingYds']])

# Calculate correlation between Yards and Points Scored (example analysis)
correlation = df[['Yards', 'PF']].corr()

# Display the correlation matrix
print("\nCorrelation between Yards and Points Scored:")
print(correlation)

# Example of exploring the correlation between rushing yards and points scored
correlation_rushing_points = df[['RushingYds', 'PF']].corr()
print("\nCorrelation between Rushing Yards and Points Scored:")
print(correlation_rushing_points)

# CSV data for passing stats
passing_stats_csv = """
Rk,Player,Age,Pos,G,GS,QBrec,Cmp,Att,Cmp%,Yds,TD,TD%,Int,Int%,1D,Succ%,Lng,Y/A,AY/A,Y/C,Y/G,Rate,QBR,Sk,Yds,Sk%,NY/A,ANY/A,4QC,GWD,Awards,Player-additional
1,Geno Smith,34,QB,17,17,10-7-0,407,578,70.4,4320,21,3.6,15,2.6,209,48.4,71,7.5,7.03,10.6,254.1,93.2,53.8,50,338,7.96,6.34,5.93,4,4,,SmitGe00
2,Jaxon Smith-Njigba,22,WR,17,16,,1,1,100.0,35,0,0.0,0,0.0,1,100.0,35,35.0,35.00,35.0,2.1,118.7,100.0,0,0,0.00,35.00,35.00,0,0,PB,SmitJa06
3,Sam Howell,24,QB,2,0,,5,14,35.7,24,0,0.0,1,7.1,1,16.7,12,1.7,-1.50,4.8,12.0,14.6,3.3,4,21,22.22,0.17,-2.33,0,0,,HoweSa00
,Team Totals,,,17,17,10-7-0,413,593,69.6,4020,21,3.5,16,2.7,211,47.6,71,7.4,6.88,10.6,236.5,91.5,,54,359,8.35,6.21,5.75,4,4,,-9999
"""

# CSV data for rushing and receiving stats
rushing_receiving_stats_csv = """
Rk,Player,Age,Pos,G,GS,Att,Yds,TD,1D,Succ%,Lng,Y/A,Y/G,A/G,Tgt,Rec,Yds,Y/R,TD,1D,Succ%,Lng,R/G,Y/G,Ctch%,Y/Tgt,Touch,Y/Tch,YScm,RRTD,Fmb,Awards,-9999
1,Jaxon Smith-Njigba,22,WR,17,16,5,26,0,1,60.0,8,5.2,1.5,0.3,137,100,1130,11.3,6,57,58.4,46,5.9,66.5,73.0,8.2,105,11.0,1156,6,1,PB,SmitJa06
2,D.K. Metcalf,27,WR,15,12,0,0,,,,0.0,0.0,108,66,992,15.0,5,35,49.1,71,4.4,66.1,61.1,9.2,66,15.0,992,5,2,,MetcDK00
3,Zach Charbonnet,23,RB,17,6,135,569,8,32,47.4,51,4.2,33.5,7.9,52,42,340,8.1,1,6,40.4,32,2.5,20.0,80.8,6.5,177,5.1,909,9,0,,CharZa00
4,Kenneth Walker III,24,RB,11,11,153,573,7,28,44.4,28,3.7,52.1,13.9,53,46,299,6.5,1,19,54.7,21,4.2,27.2,86.8,5.6,199,4.4,872,8,1,,WalkKe00
5,Tyler Lockett,32,WR,17,14,0,0,0,,,,0.0,0.0,74,49,600,12.2,2,36,56.8,37,2.9,35.3,66.2,8.1,49,12.2,600,2,0,,LockTy00
"""

# Load passing stats and rushing/receiving stats into DataFrames
passing_stats_df = pd.read_csv(StringIO(passing_stats_csv))
rushing_receiving_stats_df = pd.read_csv(StringIO(rushing_receiving_stats_csv))

# Clean the passing stats DataFrame
passing_stats_df = passing_stats_df.dropna(how="all")  # Remove completely empty rows
passing_stats_df['Yds'] = pd.to_numeric(passing_stats_df['Yds'], errors='coerce')
passing_stats_df['Cmp'] = pd.to_numeric(passing_stats_df['Cmp'], errors='coerce')
passing_stats_df['Att'] = pd.to_numeric(passing_stats_df['Att'], errors='coerce')

# Clean the rushing and receiving stats DataFrame
rushing_receiving_stats_df = rushing_receiving_stats_df.dropna(how="all")  # Remove completely empty rows
rushing_receiving_stats_df['Yds'] = pd.to_numeric(rushing_receiving_stats_df['Yds'], errors='coerce')
rushing_receiving_stats_df['Att'] = pd.to_numeric(rushing_receiving_stats_df['Att'], errors='coerce')

# Correlations between stats: example of correlation between rushing yards and points scored
correlation_rushing_points = rushing_receiving_stats_df[['Yds', 'TD']].corr()
print("\nCorrelation between Rushing Yards and Touchdowns:")
print(correlation_rushing_points)

# Calculate correlation between passing yards and touchdowns scored
correlation_passing_points = passing_stats_df[['Yds', 'TD']].corr()
print("\nCorrelation between Passing Yards and Touchdowns:")
print(correlation_passing_points)

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix for the selected columns
correlation_matrix = df[['Yards', 'PF', 'RushingYds', 'PassingYds']].corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Set the title for the heatmap
plt.title('Correlation Heatmap between Yards, PF, RushingYds, and PassingYds')

# Show the heatmap
plt.show()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent'
df[['Yards', 'PF', 'RushingYds', 'PassingYds']] = imputer.fit_transform(df[['Yards', 'PF', 'RushingYds', 'PassingYds']])

sns.pairplot(df[['Yards', 'PF', 'RushingYds', 'PassingYds']])
plt.show()

sns.histplot(df['Yards'], kde=True)
plt.show()

sns.boxplot(x=df['Yards'])
plt.show()

from sklearn.linear_model import LinearRegression

# Example: Predict 'PF' (Points Scored) based on 'Yards' and 'RushingYds'
model = LinearRegression()
model.fit(df[['Yards', 'RushingYds']], df['PF'])

# Coefficients of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Yards', 'PF', 'RushingYds', 'PassingYds']] = scaler.fit_transform(df[['Yards', 'PF', 'RushingYds', 'PassingYds']])

df.to_csv('processed_team_stats.csv', index=False)
