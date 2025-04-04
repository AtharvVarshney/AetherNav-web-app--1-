from location_data import LocationData
import time
import random
import pyttsx3
import speech_recognition as sr

class Navigation:
    def __init__(self):
        self.locations = LocationData.LOCATIONS
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.speak_and_print("Initializing Inside Navigation System...")
        self.speak_and_print("Loading house layout...")
        self.speak_and_print("System Ready!")

    def speak_and_print(self, text):
        print(text)
        self.engine.say(text)
        self.engine.runAndWait()

    def get_voice_command(self):
        with sr.Microphone() as source:
            print("\nListening for your destination...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"You said: {command}")
                return command
            except sr.UnknownValueError:
                self.speak_and_print("Sorry, I didn't catch that. Please try again.")
                return None
            except sr.RequestError:
                self.speak_and_print("Sorry, there was an error with the speech recognition service.")
                return None
            except sr.WaitTimeoutError:
                self.speak_and_print("No speech detected. Please try again.")
                return None

    def simulate_processing(self):
        processing_messages = [
            "Calculating shortest path...",
            "Analyzing house layout...",
            "Processing floor data...",
            "Mapping indoor coordinates...",
            "Checking door status...",
            "Verifying stairway access..."
        ]
        for _ in range(2):
            message = random.choice(processing_messages)
            self.speak_and_print(message)
            time.sleep(1.5)

    def get_fake_directions(self, destination):
        predefined_routes = {
            "master bedroom": [
                "Starting from main door",
                "Move 15 steps forward into the living room",
                "Walk 20 steps to reach the main stairs",
                "Climb up 15 stairs to the first floor",
                "Turn right and walk 10 steps along the landing",
                "Master bedroom entrance is on your right after 5 steps"
            ],
            "bedroom 1": [
                "Starting from current position",
                "Walk 20 steps to the main stairs",
                "Climb 15 stairs to first floor",
                "Turn left and walk 8 steps along the landing",
                "Bedroom 1 entrance is on your left after 3 steps"
            ],
            "game room": [
                "Starting navigation",
                "Walk 20 steps to the main stairs",
                "Climb 15 stairs to first floor",
                "Walk straight 12 steps along the hallway",
                "game room entrance is at the end"
            ],
            "living room": [
                "From the main door",
                "Walk 15 steps straight ahead",
                "You are now in the living room"
            ],
            "kitchen": [
                "Starting navigation",
                "Walk 15 steps through the living room",
                "Continue 10 steps past the dining room",
                "Kitchen entrance is 5 steps ahead"
            ],
            "back door": [
                "Starting route",
                "Walk 15 steps through the living room",
                "Continue 20 steps through the kitchen",
                "Back door is 5 steps ahead"
            ],
            "front lawn": [
                "Walk 5 steps to the main door",
                "Step outside onto the front lawn"
            ],
            "back lawn": [
                "Walk 15 steps through the living room",
                "Continue 20 steps through the kitchen",
                "Take 5 steps to the back door",
                "Step outside onto the back lawn"
            ],
            "game room": [
                "Starting navigation",
                "Walk 20 steps to the main stairs",
                "Climb 30 stairs to second floor",
                "Turn right and walk 10 steps at the landing",
                "Game room entrance is straight ahead"
            ],
            "home theater": [
                "From current position",
                "Walk 20 steps to the main stairs",
                "Climb 30 stairs to second floor",
                "At the landing, turn left and walk 12 steps",
                "Home theater entrance is on your left"
            ],
            "study room": [
                "Starting route",
                "Walk 20 steps to the main stairs",
                "Climb 30 stairs to second floor",
                "Turn left and walk 15 steps at the landing",
                "Study room entrance is on your left"
            ]
        }
        
        return predefined_routes.get(destination, ["Calculating route...", "Please wait..."])

    def show_available_locations(self):
        self.speak_and_print("\nAvailable locations in the house:")
        print("Ground Floor: \n- living room\n- kitchen\n- back door\n- front lawn\n- back lawn")
        print("\nFirst Floor: \n- master bedroom\n- bedroom 1\n- bedroom 2")
        print("\nSecond Floor: \n- game room\n- home theater\n- study room")
        self.speak_and_print("\nRemember these locations. You can say 'exit' anytime to quit.")

    def navigate(self):
        self.show_available_locations()
        while True:
            self.speak_and_print("\nPlease say your destination:")
            
            destination = self.get_voice_command()
            if not destination:
                continue
            
            if destination == 'exit':
                self.speak_and_print("Shutting down navigation system...")
                break
                
            if destination in self.locations:
                self.speak_and_print("\nInitiating navigation sequence...")
                self.simulate_processing()
                
                self.speak_and_print("\nGenerating route instructions:")
                for step in self.get_fake_directions(destination):
                    self.speak_and_print(f"→ {step}")
                    time.sleep(1)
                    
                self.speak_and_print("\nRoute calculated successfully!")
            else:
                self.speak_and_print("Location not found. Please try again.")
                
class output:
    def __init__(self):
        nav = Navigation()
        nav.navigate()

if __name__ == "__main__":
    output()