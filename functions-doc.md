## Data setup

|description|command|observation
--|--|--|
|Open file|read_csv()|
|Create output|.to_csv('file.csv', index=false)|
|Create dataframe|.dataframe({'column' : data})|
|Show dataframe|.head() or display()|

## Data treatment

|description|command|observation
--|--|--|
|Check NaN|.isna(row, column)|
|Count NaN|.isna().sum()|
|Substitute NaN| .fillna()|
|Drop all columns with NaN| .dropna()|

## Plotting

|description|command|observation
--|--|--|
|Plot heatmap|sn.heatmap(.corr())|
|Show heatmap| plt.show()|
|Plot scatter matrix|pd.plotting.scatter_matrix(data, c=colordata)| wont work with NaN <br> figsize, alpha, range_padding|
