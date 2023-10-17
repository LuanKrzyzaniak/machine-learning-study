# Python Machine Learning Cheatsheet

## 🐼 Pandas Dataframe
|description|command|observation
--|--|--|
|Open file|read_csv()|
|Create output|.to_csv('file.csv', index=false)|
|Create DF|pd.DataFrame(*data, rows=, columns=*)|
|Concatenate dataframes|df.concat(*df1,df2*)|
|Show DF|df or df.display()|
|Show specific lines or columns|df.iloc(*1 or 0*)[*col/line number*]|1 for columns and 0 for lines|
|Show first lines|df.head()|
|Show last lines|df.tail()|
|Find column|df.*name* or df[*name*]|
|Insert column|df[*name*] = [*data*]|
|Find line|df.loc[*line name*]|
|Insert line|df.loc[*line name*] = [*data*]|
|Find line and column|df.loc[*line*][*column*]|
|Delete line or column|df.drop(*index='name', axis=*)|index[number] also works<br>axis 0 for lines or 1 for columns|
|Atualize|df.at[*index, 'column'*] = *'name'*|we can also use df.loc|
|Stats|df.describe()|
|Count|df.count(*axis=*)|0 for item per column, 1 for attributes per line|
|Replace|df.*name*.replace([*original data*], [*new data*])|inplace=True alter the DF instead of returning a new DF|
|Order|df.sort_index(*axis=,ascending=,inline=*)|axis 1 for column, 0 for line,<br>ascending false = descending|
|Check NaN|df.isna(*row, column*)|
|Count NaN|df.isna().sum()|
|Substitute NaN| df.fillna()|
|Drop all columns with NaN| df.dropna()|
|Mean|df.mean(*axis=*)|0 for lines, 1 for column|
|Memory comsumption||
|Queries|df[][]|We can apply conditions like in SQL. Multiple conditions ask for parenthesis.|

https://colab.research.google.com/drive/1F44PLneH2NV93VPKTWxnIr3u0d7SEWrF#scrollTo=_3T9-p8DYjfZ<br>
https://docs.google.com/presentation/d/18kGOHfll7m-PEP-tvQMWxMui5TFdwdBmGCdpyLnaVh4/edit#slide=id.p

## 📊 Plotting (seaborn and pandas)
|description|command|observation
--|--|--|
|Plot heatmap|sn.heatmap(.corr())|
|Show heatmap| plt.show()|
|Plot scatter matrix|pd.plotting.scatter_matrix(data, c=colordata)| wont work with NaN <br> figsize, alpha, range_padding|
