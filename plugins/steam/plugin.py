import subprocess
import requests
import json
import re
from pluginDefault import PluginDefault

class PluginSteam(PluginDefault):

    API_KEY= "20BF4BF37BCF8EDDDD698E8EB5A5CB3A"
    
    def response(self, sentence=""):
        # Extract the game name from the input sentence
        match = re.search(r'\b(?:lance|démarre|allume)\s+(.+)\b', sentence.lower())
        if match:
            game_name = match.group(1)
        else:
            return None
        
        print(f"Okay! J'te lance {game_name}, attends j'te trouve son id...")
        # Query the Steam API for the list of apps
        url = "http://api.steampowered.com/ISteamApps/GetAppList/v0002/?key={API_KEY}&format=json"
        response = requests.get(url)
        data = json.loads(response.text)
        
        # Search for the game in the list and extract its app ID
        for app in data["applist"]["apps"]:
            if app["name"].lower() == game_name.lower():
                app_id = app["appid"]
                print(f"C'est tout bon! J'ai l'id de {game_name}!")
                # Construire la commande Steam à exécuter
                command = f'start steam://run/{app_id}'

                # Lancer la commande Steam en utilisant subprocess
                subprocess.call(command, shell=True)
                
                return f"C'est parti pour {game_name}! :D"
        
        # If the game is not found, return an error message
        return f"Désolé, je n'ai pas trouvé {game_name} dans la liste des jeux Steam. :("
