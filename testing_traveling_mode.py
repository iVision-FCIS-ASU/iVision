# # Installing necessary libraries for Offline Navigation and Geometry
# !pip install pyrosm osmnx geopy pandas
# # Downloading the latest Egypt map data (approx. 80-100MB)
# !wget https://download.geofabrik.de/africa/egypt-latest.osm.pbf

import math
import pandas as pd
import time
import os
import networkx as nx
import osmnx as ox
from pyrosm import OSM
from geopy.distance import geodesic
from shapely.geometry import Point

ox.settings.use_cache = True
ox.settings.log_console = False


class IVisionUltimateEngine:
    def __init__(self, osm_file, graph_filename="map_data/cairo_giza_walk.graphml"):
        self.osm_file = osm_file
        self.graph_file = graph_filename
        self.step_length_meters = 0.7
        self.walking_threshold_km = 1.5          # if total distance < this, always walk
        self.max_station_distance_km = 1.5       # ignore stations farther than this
        self.min_station_distance_m = 0          # ignore stations that are too close (likely same point)
        self.transit_saving_factor = 0.7         # use transit if walking to/from stations < 70% of direct walk

        self.graph_cache = {}
        self.pois_cache = {}
        self.critical_pois = pd.DataFrame()

        self._initialize_data()

    # ---------------- INIT ----------------
    def _initialize_data(self):
        start = time.perf_counter()
        key = "CairoGiza"

        print("-----Loading Graph Data-----")
        if os.path.exists(self.graph_file):
            self.graph_cache[key] = ox.load_graphml(self.graph_file)
        else:
            G = ox.graph_from_point((30.0444, 31.2357), dist=15000, network_type="walk")
            ox.save_graphml(G, self.graph_file)
            self.graph_cache[key] = G

        print("-----Loading OSM File-----")
        try:
            osm = OSM(self.osm_file)
            pois = osm.get_pois(custom_filter={
                "highway": ["bus_stop"],
                "railway": ["station", "subway_entrance", "stop"],
                "amenity": ["bus_station"],
                "station": ["subway"]
            })
            self.pois_cache[key] = pois
            self.critical_pois = pois
        except Exception as e:
            print(f"[Error Initializing OSM] {e}")

        print(f"[System] Main data ready in {round(time.perf_counter() - start, 2)} sec")

    # ---------------- MATH ----------------
    def calculate_distance(self, p1, p2):
        return geodesic(p1, p2).km

    def calculate_bearing(self, p1, p2):
        lat1, lon1 = map(math.radians, p1)
        lat2, lon2 = map(math.radians, p2)
        dLon = lon2 - lon1
        x = math.sin(dLon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
        return (math.degrees(math.atan2(x, y)) + 360) % 360

    def bearing_to_clock(self, heading, bearing):
        diff = (bearing - heading + 360) % 360
        hours = int(round(diff / 30)) or 12
        return f"{hours} o'clock"

    def get_direction(self, heading, bearing):
        diff = (bearing - heading + 360) % 360
        if diff < 20 or diff > 340:
            return "straight"
        elif diff < 70:
            return "slight right"
        elif diff < 110:
            return "right"
        elif diff < 190:
            return "turn back"
        elif diff < 290:
            return "left"
        else:
            return "slight left"

    # ---------------- GRAPH ROUTING ----------------
    def _get_nearest_node(self, lat, lon, G):
        return ox.distance.nearest_nodes(G, lon, lat)

    def _walking_directions(self, start_lat, start_lon, end_lat, end_lon, start_heading, G):
        """Generate turn‑by‑turn walking instructions using the graph."""
        try:
            start_node = self._get_nearest_node(start_lat, start_lon, G)
            end_node = self._get_nearest_node(end_lat, end_lon, G)
            route = nx.shortest_path(G, start_node, end_node, weight='length')
        except Exception as e:
            print(f"[Error] Routing failed: {e}")
            return None

        coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
        bearings = []
        for i in range(len(coords)-1):
            bearings.append(self.calculate_bearing(coords[i], coords[i+1]))

        instructions = []
        cumulative_dist = 0

        for i, (p1, p2) in enumerate(zip(coords[:-1], coords[1:])):
            seg_dist = self.calculate_distance(p1, p2) * 1000  # meters
            seg_bearing = bearings[i]

            if i == 0:
                turn_desc = "Start walking"
                ref_heading = start_heading
            else:
                prev_bearing = bearings[i-1]
                turn_desc = self.get_direction(prev_bearing, seg_bearing)
                ref_heading = prev_bearing

            clock_dir = self.bearing_to_clock(ref_heading, seg_bearing)

            instructions.append({
                'text': f"Walk {turn_desc} for {round(seg_dist)} meters. This is {clock_dir} relative to your forward direction.",
                'distance': seg_dist,
                'turn': turn_desc,
                'clock': clock_dir
            })
            cumulative_dist += seg_dist

        instructions.append({
            'text': f"You have arrived at your destination after walking {round(cumulative_dist)} meters.",
            'distance': 0,
            'turn': 'arrive',
            'clock': ''
        })
        return instructions

    def _print_walking_instructions(self, instructions):
        for i, instr in enumerate(instructions):
            if instr['turn'] == 'arrive':
                print(f"\n✓ {instr['text']}")
            else:
                print(f"  {i+1}. {instr['text']}")

    # ---------------- POI ----------------
    def find_nearest_station(self, lat, lon):
        """Find nearest station that has a name and is not extremely close."""
        if self.critical_pois is None or self.critical_pois.empty:
            return None

        df = self.critical_pois.copy()

        def get_pt_coords(g):
            if isinstance(g, Point):
                return (g.y, g.x)
            elif hasattr(g, "centroid"):
                return (g.centroid.y, g.centroid.x)
            return None

        # Add distance
        df["dist"] = df.apply(
            lambda r: self.calculate_distance((lat, lon), get_pt_coords(r.geometry))
            if get_pt_coords(r.geometry) else float('inf'), axis=1
        )

        # Filter relevant stations and require a name
        metro = df[
            ((df["railway"].isin(["station", "subway_entrance"])) |
             (df["amenity"] == "bus_station")) &
            (df["name"].notna() & (df["name"] != ""))   # require a name
        ].copy()

        # Also exclude stations that are extremely close (likely the user's own location)
        metro = metro[metro["dist"] > (self.min_station_distance_m / 1000.0)]

        # Keep only those within max distance
        metro = metro[metro["dist"] <= self.max_station_distance_km]

        metro = metro.sort_values("dist")
        if not metro.empty:
            best = metro.iloc[0]
            coords = get_pt_coords(best.geometry)
            if coords:
                return {
                    "name": best["name"],
                    "lat": coords[0],
                    "lon": coords[1],
                    "dist_to_point": best["dist"]
                }
        return None

    # ---------------- SIMPLE VOICE SIMULATION (Fallback) ----------------
    def _voice_output_simple(self, s_lat, s_lon, heading, e_lat, e_lon, target, landmarks=[]):
        """Simplified straight‑line step simulation (used when graph is not available)."""
        dist_km = self.calculate_distance((s_lat, s_lon), (e_lat, e_lon))
        if dist_km < 0.05:
            print(f"→ You have reached {target} or are very close.")
            return

        dist_m = int(dist_km * 1000)
        steps = int(dist_m / self.step_length_meters)
        bearing = self.calculate_bearing((s_lat, s_lon), (e_lat, e_lon))
        clock_dir = self.bearing_to_clock(heading, bearing)

        step_interval = 50  # simulate in chunks of ~50 steps
        current_steps = steps
        while current_steps > 0:
            display_steps = min(step_interval, current_steps)
            landmark_str = f", passing {', '.join(landmarks)}" if landmarks else ""
            print(f"→ {clock_dir}. {current_steps} steps left{landmark_str}.")
            current_steps -= display_steps

        print(f"→ Arrived at {target}")

    # ---------------- MAIN ----------------
    def execute_voice_trip(self, u_lat, u_lon, u_heading, d_lat, d_lon, d_name):
        start_time = time.perf_counter()

        # Cairo/Giza bounding box
        bbox = [29.8, 31.0, 30.5, 31.5]
        in_cairo = (bbox[0] <= u_lat <= bbox[2] and bbox[1] <= u_lon <= bbox[3] and
                    bbox[0] <= d_lat <= bbox[2] and bbox[1] <= d_lon <= bbox[3])

        # Load POIs for the area (already done for Cairo/Giza, else dynamic)
        if not (bbox[0] <= d_lat <= bbox[2] and bbox[1] <= d_lon <= bbox[3]):
            key = f"Extra_{d_name}"
            if key not in self.pois_cache:
                try:
                    osm = OSM(self.osm_file, bounding_box=[d_lat-0.1, d_lon-0.1, d_lat+0.1, d_lon+0.1])
                    self.pois_cache[key] = osm.get_pois(custom_filter={
                        "railway": ["station", "subway_entrance"],
                        "amenity": ["bus_station"]
                    })
                except Exception as e:
                    print(f"[Error loading extra POIs] {e}")
                    self.pois_cache[key] = pd.DataFrame()
            self.critical_pois = self.pois_cache[key]
        else:
            self.critical_pois = self.pois_cache.get("CairoGiza", pd.DataFrame())

        # Find stations
        boarding = self.find_nearest_station(u_lat, u_lon)
        arrival = self.find_nearest_station(d_lat, d_lon)

        total_dist = self.calculate_distance((u_lat, u_lon), (d_lat, d_lon))

        # Debug: print found stations
        print(f"[Debug] Boarding: {boarding['name'] if boarding else 'None'} at {boarding['dist_to_point']*1000:.0f} m" if boarding else "[Debug] No boarding station found")
        print(f"[Debug] Arrival: {arrival['name'] if arrival else 'None'} at {arrival['dist_to_point']*1000:.0f} m" if arrival else "[Debug] No arrival station found")

        # Decide mode
        use_transit = False
        highway_needed = False

        if total_dist > self.walking_threshold_km and not in_cairo:
            highway_needed = True
        elif total_dist > self.walking_threshold_km and boarding and arrival:
            # Calculate walking distances
            walk_to_station = boarding["dist_to_point"]
            walk_from_station = arrival["dist_to_point"]
            # Use transit only if the sum of these walks is less than a fraction of the direct walk
            if (boarding["name"] != arrival["name"] and
                (walk_to_station + walk_from_station) < self.transit_saving_factor * total_dist):
                use_transit = True

        # Summary
        print("\n=== Trip Summary ===")
        print(f"Destination: {d_name}")
        print(f"Total Distance: {round(total_dist,2)} km")
        if highway_needed:
            print("Mode: Highway Transport 🚌")
        else:
            print(f"Mode: {'Transit + Walking 🚇' if use_transit else 'Walking 🚶'}")

        # Landmarks for voice
        landmarks = []
        if boarding:
            landmarks.append(boarding["name"])
        if arrival and arrival["name"] not in landmarks:
            landmarks.append(arrival["name"])

        # Retrieve graph if in Cairo (for detailed routing)
        G = self.graph_cache.get("CairoGiza") if in_cairo else None

        if highway_needed:
            print(f"→ Use highway transport to reach {d_name}. Nearest station: {boarding['name'] if boarding else 'Unknown'}")
        elif not use_transit:
            # Walking only
            if G:
                instructions = self._walking_directions(u_lat, u_lon, d_lat, d_lon, u_heading, G)
                if instructions:
                    self._print_walking_instructions(instructions)
                else:
                    self._voice_output_simple(u_lat, u_lon, u_heading, d_lat, d_lon, d_name, landmarks)
            else:
                self._voice_output_simple(u_lat, u_lon, u_heading, d_lat, d_lon, d_name, landmarks)
        else:
            # Transit mode: walk to boarding, then transit, then walk to destination
            print(f"\n1. Walk to boarding: {boarding['name']} ({int(boarding['dist_to_point']*1000)} m)")
            if G:
                inst1 = self._walking_directions(u_lat, u_lon, boarding['lat'], boarding['lon'], u_heading, G)
                if inst1:
                    self._print_walking_instructions(inst1)
                else:
                    self._voice_output_simple(u_lat, u_lon, u_heading, boarding['lat'], boarding['lon'], boarding['name'], landmarks)
            else:
                self._voice_output_simple(u_lat, u_lon, u_heading, boarding['lat'], boarding['lon'], boarding['name'], landmarks)

            # Transit phase
            print(f"\n2. [Transit Phase]")
            print(f"✔ Enter {boarding['name']}")
            print(f"✔ Ride to {arrival['name']}")
            print(f"✔ Exit station")

            # Final walk
            print(f"\n3. Final walk to {d_name}")
            if G:
                inst2 = self._walking_directions(arrival['lat'], arrival['lon'], d_lat, d_lon, u_heading, G)
                if inst2:
                    self._print_walking_instructions(inst2)
                else:
                    self._voice_output_simple(arrival['lat'], arrival['lon'], u_heading, d_lat, d_lon, d_name, landmarks)
            else:
                self._voice_output_simple(arrival['lat'], arrival['lon'], u_heading, d_lat, d_lon, d_name, landmarks)

        print(f"\nTrip completed in {round(time.perf_counter()-start_time,2)} sec")

        return {
            "mode": "highway" if highway_needed else ("transit" if use_transit else "walking"),
            "boarding": boarding,
            "arrival": arrival,
            "landmarks": landmarks
        }


if __name__ == "__main__":
    # --- Initialize the Engine ---
    engine = IVisionUltimateEngine("map_data/egypt-latest.osm.pbf")


    # --- Test Cases ---
    test_cases = [
        
        {
            "name": "Ain Shams University",
            "u_lat": 30.15220, "u_lon": 31.33560, "u_heading": 90,
            "d_lat": 30.0700, "d_lon": 31.2830
        },
        {
            "name": "Ain Shams University",
            "u_lat": 30.0626, "u_lon": 31.2463, "u_heading": 0,
            "d_lat": 30.0700, "d_lon": 31.2830
        },
        {
            "name": "Nearby Street",
            "u_lat": 30.0440, "u_lon": 31.2354, "u_heading": 90,
            "d_lat": 30.0445, "d_lon": 31.2360
        },
        {
            "name": "Ain Shams University",
            "u_lat": 30.0626, "u_lon": 31.2463, "u_heading": 0,
            "d_lat": 30.0650, "d_lon": 31.2800
        },
        {
            "name": "Alexandria City",
            "u_lat": 31.2001, "u_lon": 29.9187, "u_heading": 180,
            "d_lat": 31.2156, "d_lon": 29.9553
        },
        {
            "name": "Remote Cairo Street",
            "u_lat": 30.0624, "u_lon": 31.2225, "u_heading": 270,
            "d_lat": 30.0650, "d_lon": 31.2300
        },
        {
            "name": "Same Spot",
            "u_lat": 30.0439, "u_lon": 31.2353, "u_heading": 45,
            "d_lat": 30.0439, "d_lon": 31.2353
        },
        {
            "name": "Helwan City Trip",  # New test case entirely in Helwan
            "u_lat": 29.8500, "u_lon": 31.3000, "u_heading": 90,
            "d_lat": 29.8600, "d_lon": 31.3150
        },
    ]

    # --- Run all test cases ---
    for i, tc in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {tc['name']} ---")
        trip_info = engine.execute_voice_trip(
            u_lat=tc["u_lat"], u_lon=tc["u_lon"], u_heading=tc["u_heading"],
            d_lat=tc["d_lat"], d_lon=tc["d_lon"], d_name=tc["name"]
        )
