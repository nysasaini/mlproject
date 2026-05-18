from django.shortcuts import render
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os

API_KEY = '7ede90bfb743fc555b59197965ca48ec'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    
    return {
        'city': data['name'],
        'main': data['weather'][0]['main'].lower(),
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'wind_gust_speed': data['wind']['speed'],
        'clouds': data['clouds']['all'],
        'visibility': data['visibility'],
        'lat': data['coord']['lat'],
        'lon': data['coord']['lon'],
        'sunrise': data['sys']['sunrise'],
        'sunset': data['sys']['sunset'],
        
    }

def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

# Prepare data for training
def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    X = data[['MinTemp', 'MaxTemp', 'WindGustDir', 'Humidity', 'Pressure', 'Temp']]
    y = data['RainTomorrow']
    return X, y, le

def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Mean Squared Error for rain Model")
    print(mean_squared_error(y_test, y_pred))
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    return X, y

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value):
    predictions = [current_value]
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

def weather_view(request):
    if request.method == "POST":
        city = request.POST.get('city')
        current_weather = get_current_weather(city)
        main_weather = current_weather['description'].split()[0].lower()
        aqi = get_air_quality(current_weather['lat'], current_weather['lon'])
        # 🔥 CONVERT SUNRISE/SUNSET
        sunrise = datetime.fromtimestamp(current_weather['sunrise']).strftime('%I:%M %p')
        sunset = datetime.fromtimestamp(current_weather['sunset']).strftime('%I:%M %p')
        # Update path to your CSV file location
        csv_path = os.path.join('C:\\Users\\diksh\\Desktop\\weather prediction\\weather.csv')  # Change this path
        historical_data = read_historical_data(csv_path)

        X, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(X, y)

        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N",   348.75, 360),
            ("N",     0,    11.25),
            ("NNE", 11.25,  33.75),
            ("NE",  33.75,  56.25),
            ("ENE", 56.25,  78.75),
            ("E",   78.75, 101.25),
            ("ESE",101.25, 123.75),
            ("SE", 123.75, 146.25),
            ("SSE",146.25, 168.75),
            ("S",  168.75, 191.25),
            ("SSW",191.25, 213.75),
            ("SW", 213.75, 236.25),
            ("WSW",236.25, 258.75),
            ("W",  258.75, 281.25),
            ("WNW",281.25, 303.75),
            ("NW", 303.75, 326.25),
            ("NNW",326.25, 348.75),
        ]

        compass_direction = next(
            (point for point, start, end in compass_points if start <= wind_deg < end),
            "N"
        )
        compass_direction_encoded = (
            le.transform([compass_direction])[0]
            if compass_direction in le.classes_ else -1
        )

        current_data = {
            'MinTemp':    current_weather['temp_min'],
            'MaxTemp':    current_weather['temp_max'],
            'WindGustDir': compass_direction_encoded,
            'Humidity':   current_weather['humidity'],
            'Pressure':   current_weather['pressure'],
            'Temp':       current_weather['current_temp'],
        }

        current_df = pd.DataFrame([current_data])
        rain_prediction = rain_model.predict(current_df)[0]
        print("RAIN PREDICTION:", rain_prediction)

        X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        X_hum,  y_hum  = prepare_regression_data(historical_data, 'Humidity')
        temp_model = train_regression_model(X_temp, y_temp)
        hum_model  = train_regression_model(X_hum,  y_hum)

        future_temp     = predict_future(temp_model, current_weather['temp_min'])
        future_humidity = predict_future(hum_model,  current_weather['humidity'])

        timezone      = pytz.timezone('Asia/Kolkata')
        now           = datetime.now(timezone)
        next_hour     = now + timedelta(hours=1)
        future_times  = [
            (next_hour + timedelta(hours=i)).strftime("%I %p").lstrip("0")
            for i in range(5)
        ]
        
        time1, time2, time3, time4, time5 = future_times
        temp1, temp2, temp3, temp4, temp5 = future_temp
        hum1, hum2, hum3, hum4, hum5 = future_humidity

        context = {
            'location': city,
            'main_weather': main_weather,
            'current_temp': current_weather['current_temp'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'description_main': current_weather['description'].split()[0].lower(),
            'city': current_weather['city'],
            'country': current_weather['country'],
            'aqi': aqi,
            'time': datetime.now().strftime("%B %d, %Y - %H:%M"),
            'date': datetime.now().strftime("%B, %d, %Y"),
            'wind': current_weather['wind_gust_speed'],  # Fixed: was pressure
            'Pressure': current_weather['pressure'],
            'visibility': round(current_weather['visibility'] / 1000, 1),
            'Maxtemp': f"{current_weather['temp_max']}°C",
            'Mintemp': f"{current_weather['temp_min']}°C",
            'sunrise': sunrise,
            'sunset': sunset,
            'lat': current_weather['lat'],
            'lon': current_weather['lon'],

            'time1': time1,
            'time2': time2,
            'time3': time3,
            'time4': time4,
            'time5': time5,

            'temp1': f"{round(temp1, 1)}",
            'temp2': f"{round(temp2, 1)}",
            'temp3': f"{round(temp3, 1)}",
            'temp4': f"{round(temp4, 1)}",
            'temp5': f"{round(temp5, 1)}",

            'hum1': f"{round(hum1, 1)}",
            'hum2': f"{round(hum2, 1)}",
            'hum3': f"{round(hum3, 1)}",
            'hum4': f"{round(hum4, 1)}",
            'hum5': f"{round(hum5, 1)}",

            'rain_prediction': "YES 🌧️" if rain_prediction == 1 else "NO ☀️",
        }

        return render(request, 'weather.html', context)
    
    # Return empty context for GET request
    return render(request, 'weather.html', {})

def get_air_quality(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    data = requests.get(url).json()

    aqi_map = {
        1: "Good",
        2: "Fair",
        3: "Moderate",
        4: "Poor",
        5: "Very Poor"
    }

    return aqi_map.get(data['list'][0]['main']['aqi'], "Unknown")

from django.http import JsonResponse

def get_weather_api(request):
    city = request.GET.get('city')

    current_weather = get_current_weather(city)

    aqi = get_air_quality(current_weather['lat'], current_weather['lon'])

    sunrise = datetime.fromtimestamp(current_weather['sunrise']).strftime('%I:%M %p')
    sunset = datetime.fromtimestamp(current_weather['sunset']).strftime('%I:%M %p')

    return JsonResponse({
        "city": current_weather['city'],
        "temp": current_weather['current_temp'],
        "description": current_weather['description'],
        "main_weather": current_weather['description'].split()[0].lower(),
        "wind": current_weather['wind_gust_speed'],
        "humidity": current_weather['humidity'],
        "aqi": aqi,
        "sunrise": sunrise,
        "sunset": sunset
    })