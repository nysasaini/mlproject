from django.shortcuts import render
from django.http import JsonResponse
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
FORECAST_URL = 'https://api.openweathermap.org/data/2.5/forecast'
ONECALL_URL  = 'https://api.openweathermap.org/data/2.5/onecall'

# ─────────────────────────────────────────────
# WEATHER FETCHERS
# ─────────────────────────────────────────────

def get_current_weather(city=None, lat=None, lon=None):
    if lat and lon:
        url = f"{BASE_URL}weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    else:
        url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    data = requests.get(url).json()
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
        'wind_gust_speed': round(data['wind']['speed'] * 3.6, 1),  # m/s → km/h
        'clouds': data['clouds']['all'],
        'visibility': data['visibility'],
        'lat': data['coord']['lat'],
        'lon': data['coord']['lon'],
        'sunrise': data['sys']['sunrise'],
        'sunset': data['sys']['sunset'],
    }


def get_10day_forecast(lat, lon):
    """
    OpenWeather free tier gives 5-day/3h forecast.
    We aggregate to daily and take up to 10 days (free = 5 days, paid = more).
    """
    url = f"{FORECAST_URL}?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&cnt=40"
    data = requests.get(url).json()

    day_map = {}
    for item in data.get('list', []):
        dt = datetime.fromtimestamp(item['dt'])
        key = dt.strftime('%Y-%m-%d')
        if key not in day_map:
            day_map[key] = {
                'temps': [],
                'descs': [],
                'mains': [],
                'rain': 0,
                'dt': dt,
            }
        day_map[key]['temps'].append(item['main']['temp'])
        day_map[key]['descs'].append(item['weather'][0]['description'])
        day_map[key]['mains'].append(item['weather'][0]['main'])
        if 'rain' in item:
            day_map[key]['rain'] += item['rain'].get('3h', 0)

    forecast_10day = []
    icon_map = {
        'Rain': 'bi-cloud-rain-fill',
        'Drizzle': 'bi-cloud-drizzle-fill',
        'Thunderstorm': 'bi-cloud-lightning-rain-fill',
        'Snow': 'bi-snow',
        'Clear': 'bi-sun-fill',
        'Clouds': 'bi-cloud-fill',
        'Mist': 'bi-cloud-fog2-fill',
        'Haze': 'bi-cloud-haze2-fill',
        'Fog': 'bi-cloud-fog-fill',
    }

    for i, (key, val) in enumerate(sorted(day_map.items())):
        if i >= 10:
            break
        dt = val['dt']
        main_cond = max(set(val['mains']), key=val['mains'].count)
        # Rain probability estimate: days with rain data
        rain_pct = min(100, round(val['rain'] * 20))
        forecast_10day.append({
            'name': 'Today' if i == 0 else dt.strftime('%a'),
            'date': dt.strftime('%d %b'),
            'icon': icon_map.get(main_cond, 'bi-cloud-sun-fill'),
            'desc': max(set(val['descs']), key=val['descs'].count).capitalize(),
            'max': round(max(val['temps'])),
            'min': round(min(val['temps'])),
            'rain': rain_pct,
        })

    return forecast_10day


def get_air_quality(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    data = requests.get(url).json()
    aqi_map = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    return aqi_map.get(data['list'][0]['main']['aqi'], "Unknown"), data['list'][0]['main']['aqi']


# ─────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────

def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.drop_duplicates()
    return df

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
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    return np.array(X).reshape(-1, 1), np.array(y)

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value):
    predictions = [current_value]
    for _ in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]


# ─────────────────────────────────────────────
# AI INSIGHT GENERATORS
# ─────────────────────────────────────────────

def get_weather_alert(temp, wind, description, aqi_num):
    """Generate a plain-language alert."""
    desc = description.lower()
    alerts = []
    if 'thunder' in desc or 'storm' in desc:
        alerts.append("⚡ Thunderstorm warning — avoid open areas and tall trees.")
        level = 'danger'
    elif 'heavy rain' in desc:
        alerts.append("🌧️ Heavy rainfall expected — potential for waterlogging and flash floods.")
        level = 'danger'
    elif 'rain' in desc:
        alerts.append("🌦️ Light to moderate rain forecast — carry an umbrella.")
        level = 'warning'
    elif temp >= 40:
        alerts.append("🌡️ Extreme heat alert — stay hydrated, avoid peak-hour sun.")
        level = 'danger'
    elif temp >= 35:
        alerts.append("☀️ High heat advisory — limit prolonged outdoor exposure.")
        level = 'warning'
    elif wind >= 50:
        alerts.append("💨 Strong wind advisory — secure loose objects outdoors.")
        level = 'warning'
    elif aqi_num >= 4:
        alerts.append("😷 Poor air quality — use masks outdoors, especially for asthma patients.")
        level = 'warning'
    else:
        alerts.append("✅ All clear! Conditions are comfortable and safe today.")
        level = 'safe'
    return ' '.join(alerts), level


def get_travel_advice(temp, description, wind):
    desc = description.lower()
    if 'thunder' in desc or 'storm' in desc:
        advice = "⛔ Avoid travel if possible. Thunderstorms can cause dangerous driving conditions."
        outdoor_score = 10
        travel_score = 20
    elif 'heavy rain' in desc:
        advice = "🚗 Drive carefully — reduced visibility and slippery roads expected."
        outdoor_score = 20
        travel_score = 40
    elif 'rain' in desc:
        advice = "🌂 Suitable for indoor travel and leisure. Plan outdoor activities post-noon."
        outdoor_score = 40
        travel_score = 65
    elif temp > 38:
        advice = "🥵 Hot weather — ideal for early morning or evening outings only."
        outdoor_score = 35
        travel_score = 60
    elif temp < 10:
        advice = "🧥 Cold weather — layer up well. Good for sightseeing if skies are clear."
        outdoor_score = 55
        travel_score = 70
    else:
        advice = "🌤️ Pleasant conditions for travel and outdoor activities today."
        outdoor_score = 85
        travel_score = 90
    return advice, outdoor_score, travel_score


def get_ai_tips(temp, description, humidity, wind, aqi_num):
    tips = []
    desc = description.lower()
    if temp >= 35:
        tips.append("Drink at least 3L of water today.")
    if 'rain' in desc or 'thunder' in desc:
        tips.append("Keep an umbrella handy at all times.")
    if aqi_num >= 3:
        tips.append("Wear an N95 mask when going outdoors.")
    if humidity >= 80:
        tips.append("High humidity — avoid strenuous workouts outside.")
    if wind >= 40:
        tips.append("Avoid riding two-wheelers at high speed today.")
    if 'clear' in desc and temp < 30:
        tips.append("Great day for a morning jog or outdoor workout!")
    if not tips:
        tips.append("Ideal weather — enjoy your day!")
    tips.append("Check back tonight for overnight forecast updates.")
    return tips[:4]


def get_disease_risks(temp, humidity, description, aqi_num):
    desc = description.lower()
    risks = []
    # Malaria / Dengue
    mosq_risk = "High" if (humidity > 75 and ('rain' in desc or temp > 28)) else "Moderate" if humidity > 60 else "Low"
    risks.append({'name': 'Dengue / Malaria', 'level': mosq_risk})
    # Respiratory
    resp_risk = "High" if aqi_num >= 4 else "Moderate" if aqi_num == 3 else "Low"
    risks.append({'name': 'Respiratory Issues', 'level': resp_risk})
    # Heat Stroke
    heat_risk = "High" if temp >= 40 else "Moderate" if temp >= 35 else "Low"
    risks.append({'name': 'Heat Stroke', 'level': heat_risk})
    # Cold / Flu
    cold_risk = "High" if temp < 10 else "Moderate" if (temp < 18 and humidity > 70) else "Low"
    risks.append({'name': 'Cold & Flu', 'level': cold_risk})
    return risks


def get_crop_advisory(temp, humidity, description, wind):
    desc = description.lower()
    if 'heavy rain' in desc or 'thunder' in desc:
        return ("Heavy rainfall expected — delay any pesticide or fertilizer application. "
                "Ensure proper drainage in fields to prevent waterlogging and root rot.")
    elif 'rain' in desc:
        return ("Moderate rain forecast — good for transplanting seedlings. "
                "Hold irrigation for the next 2–3 days. Monitor for fungal diseases.")
    elif temp >= 38 and humidity < 40:
        return ("Hot and dry conditions — irrigate crops in early morning or evening. "
                "Mulching is advised to retain soil moisture for Kharif crops.")
    elif temp >= 30 and humidity >= 70:
        return ("Warm and humid — high risk of fungal blights (especially for tomato, potato). "
                "Apply preventive fungicide and ensure good air circulation.")
    elif temp < 15:
        return ("Cool temperatures — protect frost-sensitive crops (banana, papaya) with covers. "
                "Ideal for sowing Rabi crops like wheat and mustard.")
    elif wind >= 40:
        return ("Strong winds expected — stake tall crops and check greenhouse covers. "
                "Avoid spraying any chemicals today.")
    else:
        return ("Weather conditions are favorable for most field operations. "
                "Good time for inter-cropping, weeding, and scheduled fertilization.")


# ─────────────────────────────────────────────
# COMPASS HELPER
# ─────────────────────────────────────────────

COMPASS_POINTS = [
    ("N",   348.75, 360), ("N", 0, 11.25), ("NNE", 11.25, 33.75),
    ("NE",  33.75, 56.25), ("ENE", 56.25, 78.75), ("E", 78.75, 101.25),
    ("ESE", 101.25, 123.75), ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75),
    ("S",   168.75, 191.25), ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25),
    ("WSW", 236.25, 258.75), ("W", 258.75, 281.25), ("WNW", 281.25, 303.75),
    ("NW",  303.75, 326.25), ("NNW", 326.25, 348.75),
]


# ─────────────────────────────────────────────
# SHARED CONTEXT BUILDER
# ─────────────────────────────────────────────

def build_context(current_weather, city=None):
    aqi_label, aqi_num = get_air_quality(current_weather['lat'], current_weather['lon'])
    sunrise = datetime.fromtimestamp(current_weather['sunrise']).strftime('%I:%M %p')
    sunset  = datetime.fromtimestamp(current_weather['sunset']).strftime('%I:%M %p')

    csv_path = os.path.join('C:\\Users\\diksh\\Desktop\\weather prediction\\weather.csv')
    historical_data = read_historical_data(csv_path)
    X, y, le = prepare_data(historical_data)
    rain_model = train_rain_model(X, y)

    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_direction = next(
        (point for point, start, end in COMPASS_POINTS if start <= wind_deg < end), "N"
    )
    compass_encoded = (
        le.transform([compass_direction])[0]
        if compass_direction in le.classes_ else -1
    )

    current_df = pd.DataFrame([{
        'MinTemp':    current_weather['temp_min'],
        'MaxTemp':    current_weather['temp_max'],
        'WindGustDir': compass_encoded,
        'Humidity':   current_weather['humidity'],
        'Pressure':   current_weather['pressure'],
        'Temp':       current_weather['current_temp'],
    }])
    rain_prediction = rain_model.predict(current_df)[0]

    X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    X_hum,  y_hum  = prepare_regression_data(historical_data, 'Humidity')
    temp_model = train_regression_model(X_temp, y_temp)
    hum_model  = train_regression_model(X_hum, y_hum)

    future_temp     = predict_future(temp_model, current_weather['temp_min'])
    future_humidity = predict_future(hum_model,  current_weather['humidity'])

    tz       = pytz.timezone('Asia/Kolkata')
    now      = datetime.now(tz)
    next_hr  = now + timedelta(hours=1)
    future_times = [
        (next_hr + timedelta(hours=i)).strftime("%I %p").lstrip("0")
        for i in range(5)
    ]

    # 10-day forecast
    forecast_10day = get_10day_forecast(current_weather['lat'], current_weather['lon'])

    # AI features
    weather_alert, alert_level = get_weather_alert(
        current_weather['current_temp'], current_weather['wind_gust_speed'],
        current_weather['description'], aqi_num
    )
    travel_advice, outdoor_score, travel_score = get_travel_advice(
        current_weather['current_temp'], current_weather['description'],
        current_weather['wind_gust_speed']
    )
    ai_tips = get_ai_tips(
        current_weather['current_temp'], current_weather['description'],
        current_weather['humidity'], current_weather['wind_gust_speed'], aqi_num
    )
    disease_risks = get_disease_risks(
        current_weather['current_temp'], current_weather['humidity'],
        current_weather['description'], aqi_num
    )
    crop_advice = get_crop_advisory(
        current_weather['current_temp'], current_weather['humidity'],
        current_weather['description'], current_weather['wind_gust_speed']
    )

    return {
        'location': city or current_weather['city'],
        'main_weather': current_weather['description'].split()[0].lower(),
        'current_temp': current_weather['current_temp'],
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'feels_like': current_weather['feels_like'],
        'humidity': current_weather['humidity'],
        'clouds': current_weather['clouds'],
        'description': current_weather['description'],
        'city': current_weather['city'],
        'country': current_weather['country'],
        'aqi': aqi_label,
        'aqi_num': aqi_num,
        'wind': current_weather['wind_gust_speed'],
        'Pressure': current_weather['pressure'],
        'visibility': round(current_weather['visibility'] / 1000, 1),
        'sunrise': sunrise,
        'sunset': sunset,
        'lat': current_weather['lat'],
        'lon': current_weather['lon'],
        'time1': future_times[0], 'time2': future_times[1],
        'time3': future_times[2], 'time4': future_times[3], 'time5': future_times[4],
        'temp1': round(future_temp[0], 1), 'temp2': round(future_temp[1], 1),
        'temp3': round(future_temp[2], 1), 'temp4': round(future_temp[3], 1),
        'temp5': round(future_temp[4], 1),
        'rain_prediction': "YES 🌧️" if rain_prediction == 1 else "NO ☀️",
        'forecast_10day': forecast_10day,
        # AI features
        'weather_alert': weather_alert,
        'alert_level': alert_level,
        'travel_advice': travel_advice,
        'outdoor_score': outdoor_score,
        'travel_score': travel_score,
        'ai_tips': ai_tips,
        'disease_risks': disease_risks,
        'crop_advice': crop_advice,
    }


# ─────────────────────────────────────────────
# VIEWS
# ─────────────────────────────────────────────

def weather_view(request):
    if request.method == "POST":
        city = request.POST.get('city')
        current_weather = get_current_weather(city=city)
        context = build_context(current_weather, city=city)
        return render(request, 'weather.html', context)
    return render(request, 'weather.html', {})


def weather_by_location(request):
    """
    Called by the frontend with ?lat=XX&lon=YY (from Geolocation API).
    Add to urls.py:  path('weather/location/', views.weather_by_location)
    """
    lat = request.GET.get('lat')
    lon = request.GET.get('lon')
    if not lat or not lon:
        return render(request, 'weather.html', {})
    current_weather = get_current_weather(lat=lat, lon=lon)
    context = build_context(current_weather)
    return render(request, 'weather.html', context)


def get_weather_api(request):
    city = request.GET.get('city')
    current_weather = get_current_weather(city=city)
    aqi_label, aqi_num = get_air_quality(current_weather['lat'], current_weather['lon'])
    sunrise = datetime.fromtimestamp(current_weather['sunrise']).strftime('%I:%M %p')
    sunset  = datetime.fromtimestamp(current_weather['sunset']).strftime('%I:%M %p')
    return JsonResponse({
        "city": current_weather['city'],
        "temp": current_weather['current_temp'],
        "description": current_weather['description'],
        "main_weather": current_weather['description'].split()[0].lower(),
        "wind": current_weather['wind_gust_speed'],
        "humidity": current_weather['humidity'],
        "aqi": aqi_label,
        "sunrise": sunrise,
        "sunset": sunset,
    })