import speech_recognition as sr
import pyttsx3
import numpy as np
import networkx as nx
from navigation import Navigation
from navigation import output
from location_data import LocationData

class NavigationSystem:
    def __init__(self, ply_file_path):
        self.model_data = self.load_ply_model(ply_file_path)
        self.locations = LocationData.LOCATIONS
        self.current_position = LocationData.INITIAL_POSITION
        self.graph = self.create_navigation_graph()
        self.initialize_speech_system()

    def create_navigation_graph(self):
        graph = nx.Graph()
        # Add all locations as nodes
        for location, coords in self.locations.items():
            graph.add_node(coords)
        
        # Add edges based on connected paths
        for start, end in LocationData.CONNECTED_PATHS:
            start_coords = self.locations[start]
            end_coords = self.locations[end]
            # Calculate Euclidean distance between points as edge weight
            weight = np.linalg.norm(np.array(start_coords) - np.array(end_coords))
            graph.add_edge(start_coords, end_coords, weight=weight)
        
        return graph

    def load_ply_model(self, file_path):
        try:
            ply_data = PlyData.read(file_path)
            return ply_data
        except Exception as e:
            print(f"Error loading PLY file: {e}")
            return None

    def create_navigation_graph(self):
        graph = nx.Graph()
        # Create navigation graph from PLY data
        # Add nodes and edges based on the model structure
        return graph

    def initialize_speech_system(self):
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()

    def get_voice_command(self):
        with sr.Microphone() as source:
            print("Listening for command...")
            audio = self.recognizer.listen(source)
            try:
                command = self.recognizer.recognize_google(audio)
                return command.lower()
            except sr.UnknownValueError:
                self.speak("Sorry, I didn't catch that. Please repeat.")
                return None
            except sr.RequestError:
                self.speak("Sorry, there was an error with the speech recognition service.")
                return None

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def calculate_route(self, destination):
        if destination not in self.locations:
            return None
        
        # Convert coordinates to location names for path finding
        current_location = None
        for loc, coords in self.locations.items():
            if coords == self.current_position:
                current_location = loc
                break
    
        # Calculate route using location names instead of coordinates
        try:
            path = nx.shortest_path(self.graph, 
                                  source=current_location,
                                  target=destination)
            # Convert path of location names to coordinates
            return [self.locations[loc] for loc in path]
        except nx.NetworkXNoPath:
            return None

    def create_navigation_graph(self):
        graph = nx.Graph()
        # Add all locations as nodes using location names
        for location in self.locations.keys():
            graph.add_node(location)
        
        # Add edges based on connected paths
        for start, end in LocationData.CONNECTED_PATHS:
            start_coords = self.locations[start]
            end_coords = self.locations[end]
            # Calculate Euclidean distance between points as edge weight
            weight = np.linalg.norm(np.array(start_coords) - np.array(end_coords))
            graph.add_edge(start, end, weight=weight)
        
        return graph

    def give_directions(self, destination):
        route = self.calculate_route(destination)
        if route is None:
            self.speak(f"Sorry, I cannot find a route to the {destination}")
            return

        # Convert route to user-friendly directions
        directions = self.convert_route_to_directions(route)
        for direction in directions:
            self.speak(direction)

    def convert_route_to_directions(self, route):
        directions = []
        for i in range(len(route) - 1):
            current = route[i]
            next_point = route[i + 1]
            
            # Calculate direction vector
            direction = np.array(next_point) - np.array(current)
            
            # Convert vector to cardinal directions
            angle = np.arctan2(direction[1], direction[0])
            distance = np.linalg.norm(direction)
            
            # Convert angle to compass direction
            compass_dir = self.angle_to_direction(angle)
            
            directions.append(f"Go {compass_dir} for {int(distance)} meters")
        
        return directions

    def angle_to_direction(self, angle):
        directions = ["east", "northeast", "north", "northwest", 
                     "west", "southwest", "south", "southeast"]
        index = int((angle + np.pi) / (2 * np.pi / 8))
        return directions[index % 8]

    def run(self):
        self.speak("Navigation system is ready. Where would you like to go?")
        while True:
            command = self.get_voice_command()
            if command:
                if "exit" in command or "quit" in command:
                    self.speak("Ending navigation system. Goodbye!")
                    break
                
                for location in self.locations:
                    if location in command:
                        self.speak(f"Calculating route to {location}")
                        self.give_directions(location)
                        break
                else:
                    self.speak("Sorry, I don't recognize that location")

if __name__ == "__main__":
    try:
        nav = output()
    except Exception as e:
        print("Error initializing navigation system:", str(e))