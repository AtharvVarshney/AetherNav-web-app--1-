import pyttsx3
import speech_recognition as sr
import wolframalpha
from datetime import datetime
import requests
import wikipedia
import pywhatkit as kit
from email.message import EmailMessage
import smtplib
from threading import Timer, Thread
from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
import google.generativeai as genai
import cv2
import pytesseract
import os
import numpy as np
import osmnx as ox
import networkx as nx
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import tkinter as tk
from tkinter import scrolledtext
from shapely.geometry import Point  # Added for coordinate projection

# Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize text-to-speech engine
engine = pyttsx3.init('sapi5')
engine.setProperty('volume', 1.5)
engine.setProperty('rate', 200)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Geocoder for location lookup
geolocator = Nominatim(user_agent="navigation_assistant")

# Gemini API setup
GEMINI_API_KEY = "AIzaSyBWnpUiaCGwsx8Smwx9cCP-OS4kJIiNSk0"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

# Email credentials
EMAIL = "your_email@gmail.com"
PASSWORD = "your_app_password"

# Face recognition setup
DATASET_DIR = "face_datasets"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer_cv = cv2.face.LBPHFaceRecognizer_create()
labels = {}
label_id = 0

# GUI setup
root = tk.Tk()
root.title("Voice-Controlled Virtual Assistant")
root.geometry("600x400")

# Text area for output
output_text = scrolledtext.ScrolledText(root, width=70, height=25, wrap=tk.WORD)
output_text.pack(pady=10)

def speak(text):
    """Convert text to speech and display in GUI"""
    engine.say(text)
    engine.runAndWait()
    output_text.insert(tk.END, f"Assistant: {text}\n")
    output_text.yview(tk.END)

def greet_me():
    hour = datetime.now().hour
    if 6 <= hour < 12:
        speak("Good Morning!")
    elif 12 <= hour <= 16:
        speak("Good Afternoon!")
    elif 16 <= hour < 19:
        speak("Good Evening!")
    speak("I am your personal virtual assistant powered by Gemini. Now with face detection, text reading, and outdoor navigation capabilities. Say 'help' for a list of commands.")

def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        output_text.insert(tk.END, "Listening...\n")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        output_text.insert(tk.END, "Recognizing...\n")
        query = r.recognize_google(audio, language='en-in').lower()
        output_text.insert(tk.END, f"User said: {query}\n")
        return query
    except Exception:
        speak("Sorry, I couldn't understand. Please repeat.")
        return "none"

def listen_command(timeout=7):
    with sr.Microphone() as source:
        output_text.insert(tk.END, "Listening...\n")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=7)
            command = recognizer.recognize_google(audio).lower().strip()
            output_text.insert(tk.END, f"User said: '{command}'\n")
            return command
        except (sr.UnknownValueError, sr.RequestError, sr.WaitTimeoutError):
            speak("Sorry, I didn't catch that. Please speak clearly and try again.")
            return None

# Helper Functions
def help_command():
    speak("Here are the commands you can use: 'face detection', 'text reading', 'navigation', 'ip address', 'news', 'weather', 'time and date', 'set reminder', 'volume', 'youtube', 'google', 'wikipedia', 'email', 'calculate', 'what is', 'who is', 'which is', or 'exit'. Say a command to proceed.")

# Existing Functions
def find_my_ip():
    ip_address = requests.get('https://api.ipify.org?format=json').json()
    ip = ip_address["ip"]
    speak(f"Your IP address is {ip}")
    return ip

def search_on_wikipedia(query):
    results = wikipedia.summary(query, sentences=2)
    speak(f"According to Wikipedia, {results}")
    return results

def search_on_google(query):
    kit.search(query)
    speak(f"Searching Google for {query}")

def youtube(video):
    kit.playonyt(video)
    speak(f"Playing {video} on YouTube")

def send_email(receiver_add, subject, message):
    try:
        email = EmailMessage()
        email['To'] = receiver_add
        email['Subject'] = subject
        email['From'] = EMAIL
        email.set_content(message)
        s = smtplib.SMTP("smtp.gmail.com", 587)
        s.starttls()
        s.login(EMAIL, PASSWORD)
        s.send_message(email)
        s.close()
        speak("Email sent successfully!")
        return True
    except Exception as e:
        speak("Failed to send email. Check the logs.")
        print(e)
        return False

def get_news():
    news_headline = []
    result = requests.get("https://newsapi.org/v2/top-headlines?country=us&category=general&apiKey=a2828476f7db4d368e4ed2875b1c1228").json()
    articles = result["articles"]
    for article in articles:
        news_headline.append(article["title"])
    speak("Here are today's top headlines.")
    for headline in news_headline[:3]:
        speak(headline)
    return news_headline[:3]

def weather_forecast(city):
    result = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=24194ea19bb30f888cc392eb8d524329").json()
    weather = result["weather"][0]["main"]
    temperature = result["main"]["temp"]
    feels_like = result["main"]["feels_like"]
    speak(f"In {city}, the weather is {weather}. The temperature is {temperature - 273.15:.1f} degrees Celsius, but it feels like {feels_like - 273.15:.1f} degrees Celsius.")
    return weather, f"{temperature - 273.15:.1f}°C", f"{feels_like - 273.15:.1f}°C"

def tell_time_date():
    current_time = datetime.now().strftime("%I:%M %p")
    current_date = datetime.now().strftime("%B %d, %Y")
    speak(f"The current time is {current_time} and today's date is {current_date}.")

def set_reminders():
    speak("What should I remind you about?")
    reminder_text = take_command().capitalize()
    if reminder_text == "none":
        return
    speak("In how many minutes should I remind you?")
    try:
        minutes = float(take_command())
        seconds = minutes * 60
        speak(f"I will remind you about {reminder_text} in {minutes} minutes.")
        Timer(seconds, lambda: speak(f"Reminder: {reminder_text}")).start()
    except ValueError:
        speak("Sorry, I didn't understand the time. Please try again.")

def adjust_volume():
    speak("Would you like to increase, decrease, or mute the volume?")
    command = take_command().lower()
    sessions = AudioUtilities.GetAllSessions()
    for session in sessions:
        volume = session._ctl.QueryInterface(ISimpleAudioVolume)
        current_volume = volume.GetMasterVolume()
        if "increase" in command:
            new_volume = min(1.0, current_volume + 0.1)
            volume.SetMasterVolume(new_volume, None)
            speak("Volume increased.")
        elif "decrease" in command:
            new_volume = max(0.0, current_volume - 0.1)
            volume.SetMasterVolume(new_volume, None)
            speak("Volume decreased.")
        elif "mute" in command:
            volume.SetMute(1, None)
            speak("Volume muted.")
        else:
            speak("I didn't understand. Say increase, decrease, or mute.")

def gemini_chat(query):
    try:
        response = model.generate_content(query)
        speak(response.text)
    except Exception as e:
        speak("Sorry, I couldn't get an answer from Gemini. Please try again.")
        print(f"Gemini error: {e}")

def capture_and_read_text():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Sorry, I couldn't access the camera.")
        return

    speak("Sure, I'll read the text for you. Please hold the text in front of the camera.")
    
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            speak("Failed to capture an image. Please try again.")
            cap.release()
            return

    cv2.imwrite("captured_text.jpg", frame)
    cap.release()

    try:
        image = cv2.imread("captured_text.jpg")
        text = pytesseract.image_to_string(image)
        if text.strip():
            speak("Here is the text I found:")
            speak(text)
        else:
            speak("I couldn't find any readable text in the image. Please try again with clearer text.")
    except Exception as e:
        speak("Sorry, there was an error reading the text. Please try again.")
        print(f"OCR error: {e}")

    if os.path.exists("captured_text.jpg"):
        os.remove("captured_text.jpg")

def text_reading_module():
    speak("Text reading mode activated. Say 'read text' to start, or 'exit' to return to main.")
    while True:
        query = take_command()
        if query == "none":
            continue
        if "read text" in query:
            capture_and_read_text()
        elif "exit" in query:
            speak("Returning to main assistant.")
            break
        else:
            speak("Say 'read text' to read from the camera, or 'exit' to return to main.")

def collect_dataset():
    global label_id, labels
    speak("Starting dataset collection. Please face the camera.")
    cap = cv2.VideoCapture(0)
    count = 0
    
    while count < 50:
        ret, frame = cap.read()
        if not ret:
            speak("Camera error")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100))
            temp_path = os.path.join(DATASET_DIR, f"temp_{count}.jpg")
            cv2.imwrite(temp_path, face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Collected: {count}/50", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Collecting Dataset", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if count == 50:
        speak("Dataset collection complete. Please say the name of this person.")
        name = take_command()
        if name and name != "none":
            person_dir = os.path.join(DATASET_DIR, name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
            for i in range(50):
                temp_path = os.path.join(DATASET_DIR, f"temp_{i}.jpg")
                new_path = os.path.join(person_dir, f"face_{i}.jpg")
                os.rename(temp_path, new_path)
            labels[name] = label_id
            label_id += 1
            train_recognizer()
            speak(f"Dataset saved and trained for {name}")
        else:
            speak("Name not recognized. Dataset discarded.")
            for i in range(count):
                os.remove(os.path.join(DATASET_DIR, f"temp_{i}.jpg"))
    else:
        speak("Dataset collection failed. Not enough images captured.")
        for i in range(count):
            os.remove(os.path.join(DATASET_DIR, f"temp_{i}.jpg"))

def train_recognizer():
    faces = []
    ids = []
    for person_name, person_id in labels.items():
        person_dir = os.path.join(DATASET_DIR, person_name)
        for filename in os.listdir(person_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(person_dir, filename)
                face_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(face_img)
                ids.append(person_id)
    if faces:
        recognizer_cv.train(faces, np.array(ids))
        speak("Face recognizer trained successfully")

def analyze_face():
    speak("Starting face analysis")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Camera error")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (100, 100))
            label, confidence = recognizer_cv.predict(face_roi)
            name = "Unknown"
            for n, lid in labels.items():
                if lid == label and confidence < 100:
                    name = n
                    break
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            speak(f"I see {name}")
        
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def face_detection_module():
    global labels, label_id
    label_id = 0
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_dir) and person_name not in labels:
            labels[person_name] = label_id
            label_id += 1
    train_recognizer()
    
    speak("Face detection mode activated. Say 'collect dataset' to add a new person, 'analyze face' to recognize faces, or 'exit' to return to main.")
    while True:
        command = take_command()
        if command == "none":
            continue
        if "collect dataset" in command:
            collect_dataset()
        elif "analyze face" in command:
            analyze_face()
        elif "exit" in command:
            speak("Returning to main assistant.")
            break
        else:
            speak("Say 'collect dataset' to add a person, 'analyze face' to recognize faces, or 'exit' to return to main.")

def get_current_location():
    try:
        response = requests.get('http://ip-api.com/json/', timeout=5).json()
        if response['status'] == 'success':
            lat, lon = response['lat'], response['lon']
            location = geolocator.reverse((lat, lon), language='en')
            speak(f"I have detected your current location as approximately {location.address}.")
            return lat, lon
        else:
            speak("Unable to determine your location. Please try again.")
            return None, None
    except Exception as e:
        speak("Error detecting your location.")
        print(f"Location error: {e}")
        return None, None

def get_destination():
    speak("Please tell me where you want to go.")
    destination = listen_command(timeout=7)
    if destination:
        try:
            location = geolocator.geocode(destination)
            if location:
                speak(f"Found destination: {location.address}. Please wait a moment, then say yes or no clearly.")
                time.sleep(1)
                for attempt in range(3):
                    confirmation = listen_command(timeout=7)
                    if confirmation:
                        confirmation = confirmation.lower().strip()
                        output_text.insert(tk.END, f"Confirmation attempt {attempt + 1}: '{confirmation}'\n")
                        if any(positive in confirmation for positive in ["yes", "yep", "yeah", "correct"]):
                            speak("Destination confirmed.")
                            return location.latitude, location.longitude, location.address
                        elif any(negative in confirmation for negative in ["no", "nope", "not"]):
                            speak("Let's try again.")
                            return get_destination()
                        else:
                            speak("I didn't understand that response.")
                    speak(f"Please say yes or no clearly. Attempt {attempt + 2} of 3 remaining.")
                speak("Couldn't confirm destination after 3 attempts. Let's start over.")
                return get_destination()
            else:
                speak("Couldn't find that location. Please try a different name or be more specific.")
                return get_destination()
        except Exception as e:
            speak("Error finding destination.")
            print(f"Geocode error: {e}")
            return None, None, None
    speak("No destination heard. Let's try again.")
    return get_destination()

def calculate_route(start_lat, start_lon, dest_lat, dest_lon):
    """Calculate the shortest route using OpenStreetMap with projected graph"""
    try:
        
        start_point = (start_lat, start_lon)
        dest_point = (dest_lat, dest_lon)
        distance = geodesic(start_point, dest_point).meters
        
        
        radius = min(max(distance / 2 + 1000, 1000), 10000) 
        
       
        center_lat = (start_lat + dest_lat) / 2
        center_lon = (start_lon + dest_lon) / 2
        
        
        G = ox.graph_from_point((center_lat, center_lon), dist=radius, network_type='walk')
        
        G_proj = ox.project_graph(G)
        
        
        start_proj = ox.projection.project_geometry(Point(start_lon, start_lat), to_crs=G_proj.graph['crs'])[0]
        dest_proj = ox.projection.project_geometry(Point(dest_lon, dest_lat), to_crs=G_proj.graph['crs'])[0]
        
        
        start_node = ox.distance.nearest_nodes(G_proj, start_proj.x, start_proj.y)
        dest_node = ox.distance.nearest_nodes(G_proj, dest_proj.x, dest_proj.y)
        
        
        route = nx.shortest_path(G_proj, start_node, dest_node, weight='length')
        
        
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
        return route_coords, G
    except nx.NetworkXNoPath:
        speak("No walking path found between these locations. Please try a closer destination.")
        return None, None
    except Exception as e:
        speak("Error calculating route. The area might be too large or unreachable. Please try a closer destination.")
        print(f"Route error: {e}")
        return None, None

def provide_directions(route_coords, G):
    if not route_coords or len(route_coords) < 2:
        speak("No valid route found.")
        return
    
    speak("Starting navigation. Follow my instructions carefully.")
    
    for i in range(len(route_coords) - 1):
        current = route_coords[i]
        next_point = route_coords[i + 1]
        
        distance = geodesic(current, next_point).meters
        
        lat_diff = next_point[0] - current[0]
        lon_diff = next_point[1] - current[1]
        if lat_diff > 0 and abs(lat_diff) > abs(lon_diff):
            direction = "north"
        elif lat_diff < 0 and abs(lat_diff) > abs(lon_diff):  
            direction = "south"
        elif lon_diff > 0 and abs(lon_diff) > abs(lat_diff):
            direction = "east"
        elif lon_diff < 0 and abs(lon_diff) > abs(lat_diff):
            direction = "west"
        else:
            direction = "straight"
        
        speak(f"In {int(distance)} meters, head {direction}.")
        time.sleep(min(distance / 1.4, 10))
        speak("Be cautious of potential obstacles. Keep your cane or guide ready.")
    
    speak("You have reached your destination. Navigation complete.")

def navigation_module():
    speak("Navigation module activated. Say 'guide me' to start navigation, or 'exit' to return to main.")
    while True:
        command = listen_command()
        if command and "guide me" in command:
            speak("Navigation system is now active.")
            start_lat, start_lon = get_current_location()
            if start_lat is None:
                continue
            dest_lat, dest_lon, dest_address = get_destination()
            if dest_lat is None:
                continue
            route_coords, G = calculate_route(start_lat, start_lon, dest_lat, dest_lon)
            if route_coords:
                speak(f"Guiding you from your current location to {dest_address}.")
                provide_directions(route_coords, G)
            speak("Say 'guide me' again for a new route, or 'exit' to stop.")
        elif command and "exit" in command:
            speak("Navigation module deactivated. Returning to main assistant.")
            break
        else:
            speak("Say 'guide me' to start navigation, or 'exit' to return to main.")

def voice_loop():
    while True:
        query = take_command()
        if query == "none":
            continue

        if "help" in query:
            help_command()
        elif "face detection" in query:
            face_detection_module()
        elif "text reading" in query:
            text_reading_module()
        elif "navigation" in query or "navigate" in query:
            navigation_module()
        elif "ip address" in query:
            find_my_ip()
        elif "news" in query:
            get_news()
        elif "weather" in query:
            speak("Tell me the name of your city.")
            city = take_command()
            if city != "none":
                weather_forecast(city)
        elif "time and date" in query or "tell me the time" in query or "what's the date" in query:
            tell_time_date()
        elif "set a reminder" in query:
            set_reminders()
        elif "volume" in query:
            adjust_volume()
        elif "youtube" in query:
            speak("What video would you like to watch?")
            video = take_command()
            if video != "none":
                youtube(video)
        elif "google" in query or "open google" in query:
            speak("What do you want to search on Google?")
            search_query = take_command()
            if search_query != "none":
                search_on_google(search_query)
        elif "wikipedia" in query:
            speak("What do you want to search on Wikipedia?")
            search = take_command()
            if search != "none":
                search_on_wikipedia(search)
        elif "email" in query or "send an email" in query:
            speak("To whom should I send the email? Please say the email address.")
            receiver = take_command()
            if receiver == "none":
                continue
            speak("What should be the subject?")
            subject = take_command().capitalize()
            if subject == "none":
                continue
            speak("What is the message?")
            message = take_command().capitalize()
            if message != "none":
                send_email(receiver, subject, message)
        elif "calculate" in query:
            app_id = "L5KUWU-QKYXY2XK5X"
            client = wolframalpha.Client(app_id)
            text = query.split("calculate")[1].strip()
            try:
                result = client.query(text)
                ans = next(result.results).text
                speak(f"The answer is {ans}")
            except Exception:
                speak("I couldn't calculate that. Let me ask Gemini.")
                gemini_chat(query)
        elif "what is" in query or "who is" in query or "which is" in query:
            app_id = "L5KUWU-QKYXY2XK5X"
            client = wolframalpha.Client(app_id)
            if "what is" in query:
                text = query.split("what is")[1].strip()
            elif "who is" in query:
                text = query.split("who is")[1].strip()
            elif "which is" in query:
                text = query.split("which is")[1].strip()
            else:
                text = query
            try:
                result = client.query(text)
                ans = next(result.results).text
                speak(f"The answer is {ans}")
            except Exception:
                speak("I couldn't find that with WolframAlpha. Let me ask Gemini.")
                gemini_chat(query)
        elif "exit" in query or "stop" in query:
            hour = datetime.now().hour
            if 21 <= hour < 6:
                speak("Good Night sir, take care!")
            else:
                speak("Have a good day sir!")
            root.quit()
            break
        else:
            speak("Let me check with Gemini.")
            gemini_chat(query)

if __name__ == '__main__':
    greet_me()
    Thread(target=voice_loop).start()
    root.mainloop()