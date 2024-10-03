import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv('./Data/filtered_data.csv')

x='dst_host_srv_rerror_rate'

# Group the data by 'src_bytes' and 'class' and count the occurrences
point_counts = data.groupby([x, 'class']).size().reset_index(name='count')

# Plot the data
sns.scatterplot(data=data, x=x, y='class', hue='class', legend=False)

# Show markers with sizes proportional to the count of points at each coordinate
sns.scatterplot(data=point_counts, x=x, y='class', size='count', sizes=(20, 200), marker='o', color='red', legend=False)

z=f'Visualization of {x} and Class with Point Counts'

plt.title(z)
plt.xlabel(x)
plt.ylabel('Class')

plt.show()