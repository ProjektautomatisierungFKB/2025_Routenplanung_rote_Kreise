import pyrealsense2 as rs                       # Bibliothek für die Ansteuerung der Intel RealSense Tiefenkamera
import numpy as np                              # Fundamental für Bildverarbeitung: Bilder sind hier mehrdimensionale Matrizen (Arrays)
import cv2                                      # OpenCV: Die Standard-Bibliothek für Computer Vision (Filter, Erkennung, Zeichnen)
import math                                     # Für trigonometrische Funktionen (Berechnung von Lenkwinkeln)
import time                                     # Für Zeitmessungen (wichtig für den Loop-Takt beim MQTT-Senden)
import paho.mqtt.client as mqtt                 # MQTT-Client: Ermöglicht das Senden von Befehlen an den Roboter über WLAN
import heapq                                    # "Priority Queue": Eine extrem effiziente Liste, die automatisch das kleinste Element nach vorne sortiert (für A*)

# =====================================================
# Konfiguration
# =====================================================

# ArUco Marker IDs: Dienen zur Identifikation von Roboter und Ziel
VEHICLE_ID = 4
TARGET_ID = 1

# Steuerparameter
ANGLE_THRESHOLD = 15      # Toleranzbereich in Grad: Wenn der Winkel zum Ziel kleiner ist, fährt das Auto vorwärts, sonst dreht es.
DISTANCE_THRESHOLD = 40   # Annäherungsradius in Pixeln: Wenn das Auto näher als 40px am Ziel ist, gilt es als "angekommen".
GRID_SIZE = 40            # Rasterung: Das 1280x720 Bild wird in 40x40 Pixel große Quadrate unterteilt. Größer = schneller, Kleiner = präziser.

# MQTT Konfiguration für die Kommunikation mit dem ESP32/Roboter
MQTT_BROKER_IP = "85.215.169.239"
MQTT_PORT = 1883
MQTT_TOPIC = "car/cam_vel"
MQTT_PUBLISH_INTERVAL = 0.1 # Sendehäufigkeit: Wir senden 10-mal pro Sekunde (10 Hz), um das Netzwerk nicht zu überlasten.

# Befehle: Mapping von Textbefehlen zu Integer-Werten für den Mikrocontroller
# Der ESP32 auf dem Roboter versteht nur Zahlen, daher übersetzen wir hier Text in IDs.
COMMAND_MAP = {
    "STOP": 0, "FAHRE_VOR": 1, "FAHRE_ZURUECK": 2, 
    "DREHE_LINKS": 5, "DREHE_RECHTS": 6
}

# Verbesserte Hindernis-Parameter für die Farberkennung
RED_MIN_AREA = 500        # Größenfilter: Alles unter 500 Pixeln Fläche wird als Rauschen ignoriert.
RED_MIN_CIRCULARITY = 0.6 # Formfilter: 1.0 ist ein perfekter Kreis. 0.6 akzeptiert auch leicht eierförmige/verzerrte Objekte.
OBSTACLE_PADDING = 60    # Sicherheitsabstand: Wir tun so, als wäre das Hindernis 100px größer, damit der Roboter nicht streift.

# =====================================================
# A* Path Planning
# =====================================================
# Die Klasse PathPlanner berechnet den kürzesten Weg durch ein Raster (Grid)
# unter Berücksichtigung von Hindernissen.


class PathPlanner:
    def __init__(self, width, height, grid_size):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        # Berechnung der Anzahl an Spalten und Zeilen im Raster
        # Wir teilen die Bildbreite durch die Rastergröße (Integer-Division //), um die Matrix-Dimensionen zu erhalten.
        self.cols = width // grid_size                  #Anzahl der Spalten
        self.rows = height // grid_size                 #Anzahl der Zeilen

    # Heuristik-Funktion: Schätzt die Distanz zum Ziel (Luftlinie / Euklidische Distanz)
    # Dies ist der "Kompass" des A*-Algorithmus. Er bevorzugt Wege, die geometrisch näher am Ziel liegen.
    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    # Konvertiert Pixel-Koordinaten (z.B. 640, 360) in Grid-Koordinaten (z.B. 16, 9)
    # Dies ist notwendig, da der A* auf dem groben Raster arbeitet, nicht auf einzelnen Pixeln.
    def get_grid_coords(self, pixel_coords):
        x, y = pixel_coords
        return int(x // self.grid_size), int(y // self.grid_size)

    # Konvertiert Grid-Koordinaten zurück in Pixel-Koordinaten (Mittelpunkt der Zelle)
    # Das brauchen wir, um den berechneten Pfad wieder im Bild einzeichnen zu können.
    def get_pixel_coords(self, grid_coords):
        gx, gy = grid_coords
        return int(gx * self.grid_size + self.grid_size / 2), int(gy * self.grid_size + self.grid_size / 2)

    # Hauptfunktion der Pfadplanung: Findet den Weg von start_px nach end_px um obstacles herum
    def plan(self, start_px, end_px, obstacles):
        # Umrechnung der echten Koordinaten in das Rastersystem
        start = self.get_grid_coords(start_px)
        end = self.get_grid_coords(end_px)
        
        # Erstellt ein leeres Raster (Matrix voller Nullen). 0 bedeutet "befahrbar".
        grid = np.zeros((self.rows, self.cols))
        
        # Markiert Hindernisse im Raster
        for ox, oy, r in obstacles:
            # Wir rechnen den Radius des Hindernisses + Sicherheitsabstand in Rasterzellen um.
            grid_r = int((r + OBSTACLE_PADDING) / self.grid_size)
            g_ox, g_oy = self.get_grid_coords((ox, oy))
            
            # Wir definieren ein quadratisches Fenster um das Hindernis im Grid.
            # max/min Funktionen verhindern, dass wir außerhalb des Bildrandes zeichnen (Array Index Out of Bounds).
            y_min = max(0, g_oy - grid_r)
            y_max = min(self.rows, g_oy + grid_r + 1)
            x_min = max(0, g_ox - grid_r)
            x_max = min(self.cols, g_ox + grid_r + 1)
            # Setzen der Zellen in diesem Bereich auf 1 (bedeutet "blockiert").
            grid[y_min:y_max, x_min:x_max] = 1

        # A* Algorithmus Initialisierung
        open_set = [] # Die Liste der zu prüfenden Knoten (Felder)
        # Wir nutzen einen Min-Heap, damit wir immer sofort Zugriff auf den Knoten mit den geringsten Kosten haben.
        heapq.heappush(open_set, (0, start))
        
        came_from = {} # Speichert für jedes Feld, von wo wir gekommen sind (für die Rückverfolgung des Pfades).
        
        # g_score: Die tatsächlichen Kosten vom Start bis hierher. Start->Start kostet 0.
        g_score = {start: 0}                                            # Die aktuelle Entferunung zum Startpunkt
        
        # f_score: Die geschätzten Gesamtkosten (Weg bis hier + geschätzter Restweg).
        # Zu Beginn ist das nur die Luftlinie zum Ziel.
        f_score = {start: self.heuristic(start, end)}               # Luftlinie bis zum Ziel

        while open_set:
            # Nimm den Knoten aus der Liste, der den kleinsten f_score hat (vielversprechendster Weg).
            current = heapq.heappop(open_set)[1]
            
            # Abbruchbedingung: Sind wir am Ziel angekommen?
            if current == end:
                return self.reconstruct_path(came_from, current)

            # Wir prüfen alle 8 Nachbarn (Horizontal, Vertikal, Diagonal)
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Sicherheitscheck: Liegt der Nachbar innerhalb des Grids?
                if 0 <= neighbor[0] < self.cols and 0 <= neighbor[1] < self.rows:
                    # Sicherheitscheck: Ist das Feld ein Hindernis (Wert 1)? Wenn ja, überspringen.
                    if grid[neighbor[1]][neighbor[0]] == 1: continue
                    
                    # Kostenberechnung: 
                    # Ein Schritt geradeaus kostet sqrt(0^2 + 1^2) = 1.0
                    # Ein Schritt diagonal kostet sqrt(1^2 + 1^2) = 1.41 (Wurzel 2)
                    dist = math.sqrt(dx**2 + dy**2)
                    tentative_g = g_score[current] + dist

                    # Wenn wir zu diesem Nachbarn noch nie kamen ODER der neue Weg dorthin kürzer ist als der alte:
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        # Speichere den neuen, besseren Pfad
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        # f = g (Weg bis hier) + h (Schätzung Restweg)
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, end)
                        # Füge den Nachbarn zur Liste der zu prüfenden Felder hinzu
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None # Wenn die Liste leer ist und wir das Ziel nicht erreicht haben, gibt es keinen Weg.

    # Rekonstruiert den Pfad rückwärts vom Ziel zum Start
    # Wir hangeln uns im 'came_from' Dictionary vom Ziel (current) zurück zum Vorgänger, bis wir am Start sind.
    def reconstruct_path(self, came_from, current):
        path = [self.get_pixel_coords(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self.get_pixel_coords(current))
        path.reverse() # Da wir rückwärts gesucht haben, drehen wir die Liste um.
        return path

# =====================================================
# Hilfsfunktionen & Verbesserte Erkennung
# =====================================================

# Normalisiert einen Vektor auf die Länge 1 (Einheitsvektor).
# Wichtig, um nur die Richtung zu behalten, unabhängig von der Länge.
def normalize(v):
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

# Berechnet den geometrischen Mittelpunkt zwischen zwei Punkten (x,y).
# Wird genutzt, um die Mitte der Marker-Kanten zu finden.
def midpoint(p1, p2):
    return np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])

# Kernlogik der Steuerung: Berechnet, ob gedreht oder gefahren werden muss.
def steering_command(vehicle_top, vehicle_bottom, vehicle_center, target_point):
    # Vektor 1: Wohin schaut das Auto? (Von Hinten nach Vorne)
    vehicle_dir = normalize(vehicle_top - vehicle_bottom)
    # Vektor 2: Wo ist das Ziel relativ zum Auto?
    target_dir = normalize(target_point - vehicle_center)
    
    # Skalarprodukt (Dot Product): Sagt uns, wie sehr die Vektoren in die gleiche Richtung zeigen.
    dot = np.dot(vehicle_dir, target_dir)
    # Kreuzprodukt (Cross Product in 2D): Sagt uns, ob das Ziel links oder rechts vom Auto liegt.
    cross = vehicle_dir[0] * target_dir[1] - vehicle_dir[1] * target_dir[0]
    
    # atan2 berechnet den exakten Winkel (in Bogenmaß) aus diesen beiden Werten.
    angle = math.degrees(math.atan2(cross, dot))
    # Euklidische Distanz zum Ziel berechnen.
    distance = np.linalg.norm(target_point - vehicle_center)

    cmd_key = "STOP"
    # Hierarchische Logik:
    # 1. Priorität: Ausrichtung. Wenn der Winkel zu groß ist (> 15 Grad), drehen wir auf der Stelle.
    if abs(angle) > ANGLE_THRESHOLD:
        cmd_key = "DREHE_RECHTS" if angle > 0 else "DREHE_LINKS"
    # 2. Priorität: Distanz. Wenn der Winkel stimmt, prüfen wir, ob wir noch fahren müssen.
    elif distance > DISTANCE_THRESHOLD:
        cmd_key = "FAHRE_VOR"
    
    # Rückgabe des Befehls (Text & Zahl) sowie der Telemetriedaten (Winkel & Distanz)
    return cmd_key, COMMAND_MAP[cmd_key], angle, distance

# Erkennt rote, runde Hindernisse im Bild
def detect_obstacles_robust(image):
    """
    Ersetzt HoughCircles durch Kontur-Analyse + Morphologie.
    Viel stabiler gegen Rauschen und Lichtwechsel.
    """
    # 1. Bild glätten: Median Blur ersetzt jedes Pixel durch den Median der Nachbarn.
    # Das entfernt effektiv "Salz-und-Pfeffer"-Rauschen, behält aber Kanten scharf (im Gegensatz zu GaussianBlur).
    blurred = cv2.medianBlur(image, 5)
    
    # Umwandlung in HSV Farbraum (Hue, Saturation, Value).
    # BGR (Standard) ist schlecht für Farberkennung, da Helligkeit die Farbwerte verändert.
    # HSV trennt Farbe (Hue) von Helligkeit (Value).
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # 2. Rot-Maske: Rot ist im HSV-Farbkreis speziell, da es den Übergang von 360° zu 0° bildet.
    # Deshalb brauchen wir zwei Masken: Eine für den Bereich 0-10° und eine für 170-180°.
    lower_red1 = np.array([0, 100, 50])    # Bereich 1: unteres Rot
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])  # Bereich 2: oberes Rot (geht ins Violette über)
    upper_red2 = np.array([180, 255, 255])
    
    # Wir kombinieren beide Masken mit einem logischen ODER.
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 3. Morphologie: Bereinigung der binären Maske (Schwarz/Weiß Bild)
    # Erosion & Dilatation werden kombiniert.
    # "OPEN": Entfernt kleine weiße Punkte im Hintergrund (Rauschen).
    # "CLOSE": Schließt kleine schwarze Löcher im roten Objekt.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Konturen finden: OpenCV sucht die Umrisslinien aller weißen Inseln auf der Maske.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obstacles = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt) # Berechnet die Fläche in Pixeln
        
        # Filter 1: Ist das Objekt groß genug? (Filtert Pixelfehler raus)
        if area > RED_MIN_AREA:
            # Berechnet den kleinstmöglichen Kreis, der die Kontur umschließt
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            
            # Filter 2: Rundheits-Check.
            # Wir vergleichen die Fläche der Kontur mit der Fläche eines perfekten Kreises (pi * r^2).
            # Wenn das Verhältnis (Circularity) nahe 1 ist, ist es kreisförmig.
            circle_area = math.pi * (radius ** 2)
            if circle_area == 0: continue
            circularity = area / circle_area
            
            # Wenn das Objekt rund genug ist -> Wir speichern x, y und radius als Hindernis.
            if circularity > RED_MIN_CIRCULARITY:
                obstacles.append((int(x), int(y), int(radius)))
                
    return obstacles, mask # Maske wird für Debugging-Zwecke zurückgegeben

# =====================================================
# Hauptprogramm
# =====================================================

def main():
    # MQTT Client initialisieren und im Hintergrund verbinden
    mqtt_client = mqtt.Client(client_id="pc_controller")
    mqtt_client.connect(MQTT_BROKER_IP, MQTT_PORT, keepalive=60)
    mqtt_client.loop_start() # Startet den Netzwerk-Thread

    # RealSense Kamera starten und konfigurieren
    pipeline = rs.pipeline()
    config = rs.config()
    # Farb-Stream (1280x720 Pixel, RGB-Format, 30 fps) 
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    # ArUco Marker Setup (Sammlung für 4x4 Marker (IDs 0-49)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Kompatibilitäts-Check: OpenCV hat in Version 4.7 die ArUco-API geändert.
    # Dieser Block stellt sicher, dass der Code auf alten und neuen Versionen läuft.
    try:
        # Neuere OpenCV Versionen nutzen ArucoDetector Klasse
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    except AttributeError:
        detector = None 
        print("Warnung: Alte OpenCV Version erkannt. Verwende Fallback-Modus.")

    # Pfadplaner initialisieren: Übergabe der Bilddimensionen und Rastergröße
    planner = PathPlanner(1280, 720, GRID_SIZE)

    print("Robust Navigation gestartet ESC zum Beenden")
    last_publish_time = 0

    try:
        while True:
            # Warten auf neuen Kamera-Frame (blockiert, bis Bild da ist)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue # Falls Frame leer/kaputt ist, Schleife neu starten

            # Bilddaten in Numpy Array umwandeln (Raw Data -> Matrix)
            # Das Array hat die Form (720, 1280, 3)
            # 720 Zeilen, 1280 Spalten, 3 Werte pro Pixel (Blau, Grün, Rot)
            image = np.asanyarray(color_frame.get_data())            
            
            # ===================================================
            # Erkennung und Positionierung ArUco-Marker bestimmen
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # ArUco-Erkennung funktioniert am besten auf Graustufenbildern
            corners, ids, _ = detector.detectMarkers(gray)

            vehicle_data = None
            target_center = None

            # If-Bedingung erfüllt, wenn ids != None
            if ids is not None:
                # Alle gefunden Marker durchlaufen
                for i, marker_id in enumerate(ids.flatten()):
                    c = corners[i][0] # Enthaelt die Eckpunkte der Marker
                    center = np.mean(c, axis=0) # Berechnet den geometrischen Mittelpunkt der Marker
                    
                    if marker_id == VEHICLE_ID:
                        # Auto gefunden: Ecken extrahieren fuer Ausrichtung
                        tl, tr, br, bl = c
                        # Berechnung: Mittelpunkt Vorne, Mittelpunkt Hinten, Zentrum
                        top_mid = midpoint(tl, tr)
                        bottom_mid = midpoint(bl, br) 
                        vehicle_data = (top_mid, bottom_mid, center)
                        # Visualisierung: Güne Box um Fahrzeug
                        cv2.polylines(image, [c.astype(np.int32)], True, (0, 255, 0), 2)
                        
                    elif marker_id == TARGET_ID:
                        # Ziel gefunden
                        target_center = center
                        # Visualisierung: Zeichne blauen Kreis am Ziel
                        cv2.circle(image, tuple(target_center.astype(int)), 10, (255, 0, 0), -1)

            # =======================================
            # Erkennung der Hindernisse (Rote Kreise)
            obstacles, debug_mask = detect_obstacles_robust(image)
            
            # Visualisierung aller gefundenen Hindernisse ins Bild
            for ox, oy, r in obstacles:
                cv2.circle(image, (ox, oy), r, (0, 0, 255), 2) # Roter Kreis (das eigentliche Objekt)
                cv2.circle(image, (ox, oy), r + OBSTACLE_PADDING, (0, 0, 100), 1) # Dünnerer Kreis (Sicherheitszone)
                cv2.putText(image, "Obstacle", (ox - 20, oy - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

            # ====================================================
            # Bestimmen der Steuerbefehle und die optimale Pfadplanung
            cmd_str, cmd_int, angle, distance = "STOP", 0, 0, 0

            # Nur navigieren, wenn Auto UND Ziel erkannt wurden
            if vehicle_data and target_center is not None:
                top, bottom, vehicle_center = vehicle_data
                
                # Pfad berechnen: Der A* Algorithmus liefert eine Liste von Koordinaten zurück.
                path = planner.plan(vehicle_center, target_center, obstacles)
                nav_target = target_center 

                if path and len(path) > 1:
                    # Zeichne den berechneten Pfad als gelbe Linie
                    points = np.array(path, np.int32)
                    cv2.polylines(image, [points], False, (0, 255, 255), 2)
                    
                    # Waehle den naechsten Punkt auf dem Pfad als Zwischenziel
                    target_idx = 1
                    # Auto sehr nah am nächsten Punkt --> nimm uebernaesten Punkt
                    if len(path) > 2 and np.linalg.norm(np.array(path[1]) - vehicle_center) < DISTANCE_THRESHOLD:
                           target_idx = 2
                    nav_target = np.array(path[target_idx])
                    
                    # Zeichne den Punkt grün, den das Auto aktuell aktiv ansteuert
                    cv2.circle(image, tuple(nav_target), 6, (0, 255, 0), -1)

                # Steuerbefehl berechnen: Vergleich der Vektoren
                cmd_str, cmd_int, angle, distance = steering_command(top, bottom, vehicle_center, nav_target)
                # Visualisert orangenen Pfeil für Bewegungsrichtung
                cv2.arrowedLine(image, tuple(vehicle_center.astype(int)), tuple(nav_target.astype(int)), (255, 150, 0), 3)
                
            # ====================================================================
            # Steuerbefehle an Raspberry Pi senden
            # Sende den Befehl via MQTT, aber limitiert durch MQTT_PUBLISH_INTERVAL
            curr_time = time.time()
            if curr_time - last_publish_time >= MQTT_PUBLISH_INTERVAL:
                mqtt_client.publish(MQTT_TOPIC, str(cmd_int))
                last_publish_time = curr_time

            # Statusanzeige im Fenster
            cv2.putText(image, f"CMD: {cmd_str}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Navigation Robust", image)
            
            # Beenden mit ESC-Taste (ASCII 27)
            if cv2.waitKey(1) == 27: break

    finally:
        # Aufraeumen beim Beenden
        mqtt_client.loop_stop()
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
