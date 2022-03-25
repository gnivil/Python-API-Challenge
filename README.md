# Python APIs Challenge - What's the Weather Like?

-----

# Part I - WeatherPy

Created a Python script to visualize the weather of 500+ cities across the world of varying distance from the equator. Accomplished this by utilizing a simple Python library and OpenWeatherMap API to create a representative model of weather across world cities.

-----

## Set up jupyter notebook 

```python
# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import time
from scipy.stats import linregress

# Import API key
from config import weather_api_key

# Incorporate citipy to determine city based on latitude and longitude
from citipy import citipy

# Output File (CSV)
output_data_file = "output_data/cities.csv"

# Range of latitudes and longitudes
lat_range = (-90, 90)
lng_range = (-180, 180)
```

## Generate Cities List

```python
# List for holding lat_lngs and cities
lat_lngs = []
cities = []

# Create a set of random lat and lng combinations
lats = np.random.uniform(lat_range[0], lat_range[1], size=1500)
lngs = np.random.uniform(lng_range[0], lng_range[1], size=1500)
lat_lngs = zip(lats, lngs)

# Identify nearest city for each lat, lng combination
for lat_lng in lat_lngs:
    city = citipy.nearest_city(lat_lng[0], lat_lng[1]).city_name
    
    # If the city is unique, then add it to a our cities list
    if city not in cities:
        cities.append(city)

# Print the city count to confirm sufficient count
len(cities)
```

## Perform API Calls

```python
# Starting URL for Weather Map API Call
#url = "http://api.openweathermap.org/data/2.5/weather?units=Imperial&APPID=" + weather_api_key
url = "http://api.openweathermap.org/data/2.5/weather?units=Imperial&APPID=" + weather_api_key

# List of city data
city_data = []

# Print to logger
print("Beginning Data Retrieval     ")
print("-----------------------------")

# Create counters
record_count = 1
set_count = 1

# Loop through all the cities in list
for i, city in enumerate(cities):
        
    # Group cities in sets of 50 for logging purposes
    if (i % 50 == 0 and i >= 50):
        set_count += 1
        record_count = 0

    # Create endpoint URL with each city
    city_url = url + "&q=" + city
    
    # Log the url, record, and set numbers
    print("Processing Record %s of Set %s | %s" % (record_count, set_count, city))

    # Add 1 to the record count
    record_count += 1

    # Run an API request for each of the cities
    try:
        # Parse the JSON and retrieve data
        city_weather = requests.get(city_url).json()

        # Parse out the max temp, humidity, and cloudiness
        city_lat = city_weather["coord"]["lat"]
        city_lng = city_weather["coord"]["lon"]
        city_max_temp = city_weather["main"]["temp_max"]
        city_humidity = city_weather["main"]["humidity"]
        city_clouds = city_weather["clouds"]["all"]
        city_wind = city_weather["wind"]["speed"]
        city_country = city_weather["sys"]["country"]
        city_date = city_weather["dt"]

        # Append the City information into city_data list
        city_data.append({"City": city, 
                          "Lat": city_lat, 
                          "Lng": city_lng, 
                          "Max Temp": city_max_temp,
                          "Humidity": city_humidity,
                          "Cloudiness": city_clouds,
                          "Wind Speed": city_wind,
                          "Country": city_country,
                          "Date": city_date})

    # If an error is experienced, skip the city
    except:
        print("City not found. Skipping...")
        pass
              
# Indicate that Data Loading is complete 
print("-----------------------------")
print("Data Retrieval Complete      ")
print("-----------------------------")
```

## Create and Display DataFrames

```python
# Convert array of JSONs into Pandas DataFrame
city_data_pd = pd.DataFrame(city_data)

# Show Record Count
city_data_pd.count() 

# Display city_data DataFrame
city_data_pd.head()
```

## Inspect data and remove cities where humidity > 100%
Skip this step if no cities have humidity > 100%

```python
city_data_pd.describe()

#  Get the indices of cities that have humidity over 100%
dirty_city_data = city_data_pd[(city_data_pd["Humidity"] > 100)].index                  
dirty_city_data

# Make a new DataFrame equal to city_data to drop all humidity outliers by index
# Passing "inplace=False" will make a copy of the city_data DataFrame, which will be called "clean_city_data".
clean_city_data = city_data_pd.drop(dirty_city_data, inplace=False)
clean_city_data.head()

# Extract relevant fields from the DataFrame
lats = clean_city_data["Lat"]
max_temps = clean_city_data["Max Temp"]
humidity = clean_city_data["Humidity"]
cloudiness = clean_city_data["Cloudiness"]
wind_speed = clean_city_data["Wind Speed"]

# Export the city_data into a csv
clean_city_data.to_csv(output_data_file, index_label="City_ID")
```

-----

## Created a series of scatter plots showcasing various relationships with Latitude

-----

## Temperature (F) vs. Latitude
```python
# Build scatter plot for latitude vs. temperature
plt.scatter(lats, 
            max_temps,
            edgecolor="black", linewidths=1, marker="o", 
            alpha=0.8, label="Cities")

# Incorporate the other graph properties
plt.title("City Latitude vs. Max Temperature (%s)" % time.strftime("%x"))
plt.ylabel("Max Temperature (F)")
plt.xlabel("Latitude")
plt.grid(True)

# Save the figure 
plt.savefig("output_data/Fig1.png")

# Show plot
plt.show()
```
## Humidity (%) vs. Latitude
```python
# Build the scatter plots for latitude vs. humidity
plt.scatter(lats, 
            humidity,
            edgecolor="black", linewidths=1, marker="o", 
            alpha=0.8, label="Cities")

# Incorporate the other graph properties
plt.title("City Latitude vs. Humidity (%s)" % time.strftime("%x"))
plt.ylabel("Humidity (%)")
plt.xlabel("Latitude")
plt.grid(True)

# Save the figure
plt.savefig("output_data/Fig2.png")

# Show plot
plt.show()
```
## Cloudiness (%) vs. Latitude
```python
# Build the scatter plots for latitude vs. cloudiness
plt.scatter(lats, 
            cloudiness,
            edgecolor="black", linewidths=1, marker="o", 
            alpha=0.8, label="Cities")

# Incorporate the other graph properties
plt.title("City Latitude vs. Cloudiness (%s)" % time.strftime("%x"))
plt.ylabel("Cloudiness (%)")
plt.xlabel("Latitude")
plt.grid(True)

# Save the figure
plt.savefig("output_data/Fig3.png")

# Show plot
plt.show()
```
## Wind Speed (mph) vs. Latitude
```python
# Build the scatter plots for latitude vs. wind speed
plt.scatter(lats, 
            wind_speed,
            edgecolor="black", linewidths=1, marker="o", 
            alpha=0.8, label="Cities")

# Incorporate the other graph properties
plt.title("City Latitude vs. Wind Speed (%s)" % time.strftime("%x"))
plt.ylabel("Wind Speed (mph)")
plt.xlabel("Latitude")
plt.grid(True)

# Save the figure
plt.savefig("output_data/Fig4.png")

# Show plot
plt.show()
```

-----

## Linear Regression
Ran a linear regression on each regression. Seperated the plots into Northern Hemisphere (greater than or equal to 0 degrees latitude) and Souther Hemisphere (less than 0 degrees latitude): 
* Northern Hemisphere - Temperature (F) vs. Latitude
* Southern Hemisphere - Temperature (F) vs. Latitude
* Northern Hemisphere - Humidity (%) vs. Latitude
* Southern Hemisphere - Humidity (%) vs. Latitude
* Northern Hemisphere - Cloudiness (%) vs. Latitude
* Southern Hemisphere - Cloudiness (%) vs. Latitude
* Northern Hemisphere - Wind Speed (mph) vs. Latitude
* Southern Hemisphere - Wind Speed (mph) vs. Latitude

```python
# Create a function to create Linear Regression plots
def plot_linear_regression(x_values, y_values, title, text_coordinates):
    
    # Run regresson on southern hemisphere
    (slope, intercept, rvalue, pvalue, stderr) = linregress(x_values, y_values)
    regress_values = x_values * slope + intercept
    line_eq = "y = " + str(round(slope,2)) + "x + " + str(round(intercept,2))

    # Plot
    plt.scatter(x_values,y_values)
    plt.plot(x_values,regress_values,"r-")
    plt.annotate(line_eq,text_coordinates,fontsize=15,color="red")
    plt.xlabel('Latitude')
    plt.ylabel(title)
    print(f"r-value: {rvalue**2}")
    plt.show()

# Create Northern and Southern Hemisphere DataFrames
northern_hemi_df = city_data_pd.loc[(city_data_pd["Lat"] >= 0)]
southern_hemi_df = city_data_pd.loc[(city_data_pd["Lat"] < 0)]
```

-----

## Northern Hemisphere - Temperature (F) vs. Latitude

```python
# Linear regression on Northern Hemisphere
x_values = northern_hemi_df["Lat"]
y_values = northern_hemi_df["Max Temp"]
plot_linear_regression(x_values, y_values, 'Max Temp',(6,30))
```

## Southern Hemisphere - Temperature (F) vs. Latitude

```python
# Linear regression on Southern Hemisphere
x_values = southern_hemi_df["Lat"]
y_values = southern_hemi_df["Max Temp"]
plot_linear_regression(x_values, y_values, 'Max Temp', (-55, 90))
```

## Northern Hemisphere - Humidity (%) vs. Latitude

```python
# Northern Hemisphere
x_values = northern_hemi_df["Lat"]
y_values = northern_hemi_df["Humidity"]
plot_linear_regression(x_values, y_values, 'Humidity',(40,10))
```

## Southern Hemisphere - Humidity (%) vs. Latitude

```python
# Southern Hemisphere
x_values = southern_hemi_df["Lat"]
y_values = southern_hemi_df["Humidity"]
plot_linear_regression(x_values, y_values, 'Humidity', (-50, 20))
```

## Northern Hemisphere - Cloudiness (%) vs. Latitude

```python
# Northern Hemisphere
x_values = northern_hemi_df["Lat"]
y_values = northern_hemi_df["Cloudiness"]
plot_linear_regression(x_values, y_values, 'Cloudiness', (40,10))
```

## Southern Hemisphere - Cloudiness (%) vs. Latitude

```python
# Southern Hemisphere
x_values = southern_hemi_df["Lat"]
y_values = southern_hemi_df["Cloudiness"]
plot_linear_regression(x_values, y_values, 'Cloudiness', (-30,30))
```

## Northern Hemisphere - Wind Speed (mph) vs. Latitude

```python
# Northern Hemisphere
x_values = northern_hemi_df["Lat"]
y_values = northern_hemi_df["Wind Speed"]
plot_linear_regression(x_values, y_values, 'Wind Speed', (40,25))
```

## Southern Hemisphere - Wind Speed (mph) vs. Latitude

```python
# Southern Hemisphere
x_values = southern_hemi_df["Lat"]
y_values = southern_hemi_df["Wind Speed"]
plot_linear_regression(x_values, y_values, 'Wind Speed', (-50, 20))
```
-----

# Part II - VacationPy

Used OpenWeatherMap API data, jupyter-gmaps, and Google Places API to plan future vacations. 

-----

## Set up jupyter notebook

```python
# Dependencies and Setup
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import gmaps
import os

# Import API key
from config import g_key
```

## Created a heat map that displays the humidity for every city from Part I
* Loaded the cities.csv exported in WeatherPy to a DataFrame
* Configured gmaps
* Used the Lat and Lng as locations and Humidity as weight
* Added Heatmap layer to map

```python
# Store csv created in WeatherPy into a DataFrame
city_data_df = pd.read_csv("output_data/cities.csv")
city_data_df.head()

# Configure gmaps
gmaps.configure(api_key=g_key)

# Heatmap of humidity
locations = city_data_df[["Lat", "Lng"]]
humidity = city_data_df["Humidity"]
fig = gmaps.figure()
heat_layer = gmaps.heatmap_layer(locations, weights=humidity, dissipating=False, max_intensity=300, point_radius=5)

fig.add_layer(heat_layer)
fig
```

## Narrowed down DataFrame to find ideal vacation location
* Created new DataFrame fitting weather criteria (temparature between 70 an 80 degrees farenheit, wind speed less than 10mph, zero cloudiness)
* Dropped any rows that did not meet all three conditions
* Dropped any rows with null values

```python
# Narrow down cities that fit criteria and drop any results with null values
narrowed_city_df = city_data_df.loc[(city_data_df["Max Temp"] < 80) & (city_data_df["Max Temp"] > 70) \
                                    & (city_data_df["Wind Speed"] < 10) \
                                    & (city_data_df["Cloudiness"] == 0)].dropna()
narrowed_city_df
```

## Created hotel map of hotels near the selected cities
* Used Google Places API to find each city's coordinates
* Found the first hotel within 5000 meters of set coordinates
* Added "Hotel Name" column to the DataFrame
* Stored the first Hotel result into the DataFrame
* Plotted markers on to of the heatmap

```python
# Create DataFrame called hotel_df to store hotel names along with city, country and coordinates
hotel_df = narrowed_city_df[["City", "Country", "Lat", "Lng"]].copy()
hotel_df["Hotel Name"] = ""
hotel_df

# Set parameters to search for a hotel
params = {
    "radius": 5000,
    "types": "lodging",
    "key": g_key
}

# Iterate through 
for index, row in hotel_df.iterrows():
    # get lat, lng from df
    lat = row["Lat"]
    lng = row["Lng"]
    
    params["location"] = f"{lat},{lng}"
    
    # Use the search term: "Hotel" and our lat/lng
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

    # make request and print url
    name_address = requests.get(base_url, params=params)
    
    # convert to json
    name_address = name_address.json()
    
    # Grab the first hotel from the results and store the name
    try:
        hotel_df.loc[index, "Hotel Name"] = name_address["results"][0]["name"]
    except (KeyError, IndexError):
        print("Missing field/result... skipping.")

hotel_df

# Using the template add the hotel marks to the heatmap
info_box_template = """
<dl>
<dt>Name</dt><dd>{Hotel Name}</dd>
<dt>City</dt><dd>{City}</dd>
<dt>Country</dt><dd>{Country}</dd>
</dl>
"""
# Store the DataFrame Row
hotel_info = [info_box_template.format(**row) for index, row in hotel_df.iterrows()]
locations = hotel_df[["Lat", "Lng"]]

# Add marker layer ontop of heat map
marker_layer = gmaps.marker_layer(locations, info_box_content=hotel_info)
fig.add_layer(marker_layer)

# Display figure
fig
```
