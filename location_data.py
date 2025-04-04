class LocationData:
    # Initial position (front door)
    INITIAL_POSITION = (0, 0, 0)

    # Dictionary containing all locations and their coordinates (x, y, z)
    LOCATIONS = {
        # Ground Floor
        "front door": (0, 0, 0),
        "back door": (20, 0, 0),
        "main hall": (10, 10, 0),
        "reception": (5, 2, 0),
        "ground floor bathroom": (15, 5, 0),
        "elevator": (12, 8, 0),
        "main stairs": (8, 8, 0),
        
        # Second Floor (z = 5 represents the height of one floor)
        "second floor landing": (8, 8, 5),
        "conference room": (15, 15, 5),
        "second floor bathroom": (15, 5, 5),
        "study room": (20, 10, 5),
        "library": (10, 20, 5),
        
        # Third Floor (z = 10)
        "third floor landing": (8, 8, 10),
        "observation deck": (10, 10, 10),
        "storage room": (15, 15, 10),
        "third floor bathroom": (15, 5, 10),
        "archive room": (20, 20, 10)
    }

    # Define connected paths for navigation
    CONNECTED_PATHS = [
        ("front door", "reception"),
        ("reception", "main hall"),
        ("main hall", "back door"),
        ("main hall", "ground floor bathroom"),
        ("main hall", "elevator"),
        ("main hall", "main stairs"),
        ("main stairs", "second floor landing"),
        ("second floor landing", "conference room"),
        ("second floor landing", "study room"),
        ("second floor landing", "library"),
        ("second floor landing", "second floor bathroom"),
        ("second floor landing", "main stairs"),
        ("main stairs", "third floor landing"),
        ("third floor landing", "observation deck"),
        ("third floor landing", "storage room"),
        ("third floor landing", "archive room"),
        ("third floor landing", "third floor bathroom")
    ]