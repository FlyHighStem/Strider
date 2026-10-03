import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from supabase import create_client

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
if not supabase:
    print("WARNING: no Supabase credentials in .env, routes will ignore hazard reports.")

# ---- Routing config (keep CENTER in sync with CAMPUS_CENTER in static/index.html) ----
CENTER = (37.8719, -122.2585)  # (lat, lng)
RADIUS_M = 3000  # keep equal to ROUTE_RADIUS_M in static/index.html
GRAPH_FILE = BASE_DIR / "campus_walk.graphml"

LIGHT_W = 1.5      # a fully unlit segment costs up to 2.5x its length
PHONE_W = 0.5      # extra cost for being 300m+ from an emergency phone
HAZARD_W = 3.0     # extra cost per active hazard report near a segment (max 2 count)
LAMP_RANGE_M = 30
HAZARD_RANGE_M = 40
WALK_M_PER_MIN = 80
TRAFFIC_W = 1.0    # extra cost for quiet segments, scaled up as the day gets quieter
POI_RANGE_M = 100  # how far a cafe, library, bus stop, etc. counts toward a segment

# Foot traffic is an ESTIMATE, not measured counts: nearby places people go (cafes, libraries,
# dorms, bus stops) + road type, scaled by how busy campus is at that hour (index = hour, 0-23).
HOUR_ACTIVITY = [.05, .04, .03, .03, .04, .08, .2, .4, .7, .9, 1, 1, 1, 1, 1, 1, 1,
                 .95, .8, .65, .5, .35, .2, .1]
ROAD_ACTIVITY = {"primary": .9, "secondary": .9, "tertiary": .85, "pedestrian": .8, "living_street": .7,
                 "residential": .6, "unclassified": .6, "footway": .4, "cycleway": .4, "path": .35,
                 "steps": .3, "service": .25, "track": .2}
POI_TAGS = {
    "amenity": ["cafe", "restaurant", "fast_food", "bar", "pub", "library", "university", "college",
                "bank", "pharmacy", "cinema", "theatre", "food_court", "marketplace", "bus_station"],
    "shop": True,
    "highway": "bus_stop",
    "building": ["university", "dormitory", "college"],
}

ox.settings.useful_tags_way = list(ox.settings.useful_tags_way) + ["lit"]

_lock = threading.Lock()
_state = {}


def _points(tags, crs):
    """Download point features (lamps, emergency phones) near campus."""
    try:
        gdf = ox.features_from_point(CENTER, tags=tags, dist=RADIUS_M)
        gdf = gdf[gdf.geometry.type == "Point"]
        return gdf.to_crs(crs)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=crs)


def _centroids(tags, crs):
    """Download features of any shape (node or building outline) and reduce each to a point."""
    try:
        gdf = ox.features_from_point(CENTER, tags=tags, dist=RADIUS_M).to_crs(crs)
        return gdf.assign(geometry=gdf.geometry.centroid)
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs=crs)


def _latlng(gdf):
    names = gdf["name"] if "name" in gdf.columns else [None] * len(gdf)
    return [{"lat": p.y, "lng": p.x, "name": n if isinstance(n, str) else None}
            for p, n in zip(gdf.geometry.to_crs(4326), names)]


def _count_near(edges, pts, meters):
    if len(pts) == 0:
        return np.zeros(len(edges))
    hits = pts.sindex.query(edges.geometry.buffer(meters), predicate="intersects")
    return np.bincount(hits[0], minlength=len(edges)).astype(float)


def _load():
    if GRAPH_FILE.exists():
        G = ox.load_graphml(GRAPH_FILE)
    else:
        print("Downloading campus walking network (first run only)...")
        G = ox.graph_from_point(CENTER, dist=RADIUS_M, network_type="walk")
        ox.save_graphml(G, GRAPH_FILE)

    edges = ox.graph_to_gdfs(G, nodes=False)
    edges = ox.projection.project_gdf(edges)
    crs = edges.crs

    lamps = _points({"highway": "street_lamp"}, crs)
    phones = _points({"emergency": "phone"}, crs)
    police = _centroids({"amenity": "police"}, crs)
    print(f"Loaded {len(edges)} segments, {len(lamps)} lamps, {len(phones)} emergency phones, "
          f"{len(police)} police stations.")

    # Lighting: trust the OSM `lit` tag when present, otherwise infer from nearby lamps
    lamp_light = np.minimum(_count_near(edges, lamps, LAMP_RANGE_M) / 2, 1.0)
    lit = edges["lit"].astype(str).str.lower() if "lit" in edges.columns else pd.Series("nan", index=edges.index)
    light = np.where(lit == "no", 0.0, np.where(lit.isin(["nan", "none", ""]), lamp_light, 1.0))

    help_pts = pd.concat([phones.geometry, police.geometry])  # emergency phones + police stations
    if len(help_pts):
        phone_dist = edges.geometry.distance(help_pts.union_all()).to_numpy()
        phone_pen = np.clip(phone_dist / 300, 0, 1) * PHONE_W
    else:
        phone_pen = np.zeros(len(edges))

    # Baseline foot traffic: places people go nearby (60%) + how main the road is (40%)
    try:
        pois = ox.features_from_point(CENTER, tags=POI_TAGS, dist=RADIUS_M).to_crs(crs)
        pois = gpd.GeoDataFrame(geometry=pois.geometry.centroid, crs=crs)
    except Exception:
        pois = gpd.GeoDataFrame(geometry=[], crs=crs)
    poi_n = _count_near(edges, pois, POI_RANGE_M)
    scale = np.percentile(poi_n[poi_n > 0], 90) if (poi_n > 0).any() else 1
    poi_score = np.clip(poi_n / scale, 0, 1)
    road_score = edges["highway"].map(
        lambda h: max([v for k, v in ROAD_ACTIVITY.items() if k in str(h)] or [0.5])).to_numpy()
    print(f"Found {len(pois)} places that attract foot traffic.")

    edges["light"] = light
    edges["static_pen"] = (1 - light) * LIGHT_W + phone_pen
    edges["traffic_base"] = 0.6 * poi_score + 0.4 * road_score
    _state.update(G=G, edges=edges, crs=crs, phones=_latlng(phones), police=_latlng(police))


def _hazards(crs):
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_crs(crs)
    if not supabase:
        return empty
    now = datetime.now(timezone.utc).isoformat()
    rows = supabase.table("hazard_reports").select("lat,lng").gt("expires_at", now).execute().data
    if not rows:
        return empty
    pts = gpd.points_from_xy([r["lng"] for r in rows], [r["lat"] for r in rows])
    return gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326").to_crs(crs)


def _summarize(G, path, weight):
    length = lit = traffic = haz_segments = 0.0
    for u, v in zip(path[:-1], path[1:]):
        e = min(G[u][v].values(), key=lambda d: d[weight])
        length += e["length"]
        lit += e["length"] * e["light"]
        traffic += e["length"] * e["traffic"]
        haz_segments += 1 if e["haz"] > 0 else 0
    return {
        "length_m": round(length),
        "minutes": max(1, round(length / WALK_M_PER_MIN)),
        "lit_pct": round(100 * lit / length) if length else 100,
        "traffic_pct": round(100 * traffic / length) if length else 0,
        "hazard_segments": int(haz_segments),
        "coords": [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in path],
    }


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_load, daemon=True).start()  # warm up so the first click is fast
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/health")
def health():
    return {"status": "ok", "database": supabase is not None, "routing_ready": "G" in _state}


@app.get("/api/safe-points")
def safe_points():
    if "G" not in _state:
        raise HTTPException(503, "Map data is still loading.")
    return {"phones": _state["phones"], "police": _state["police"]}


@app.get("/api/route")
def route(from_lat: float, from_lng: float, to_lat: float, to_lng: float, hour: int = -1):
    if "G" not in _state:
        raise HTTPException(503, "Map data is still loading. Try again in a minute.")
    if hour not in range(24):
        hour = datetime.now().hour
    with _lock:
        G, edges, crs = _state["G"], _state["edges"], _state["crs"]

        haz = np.minimum(_count_near(edges, _hazards(crs), HAZARD_RANGE_M), 2)
        # Quiet segments cost more, and the penalty grows as fewer people are around
        activity = HOUR_ACTIVITY[hour]
        base = edges["traffic_base"].to_numpy()
        traffic_pen = TRAFFIC_W * (1 - base) * (0.3 + 0.7 * (1 - activity))
        cost = edges["length"].to_numpy() * (1 + edges["static_pen"].to_numpy() + traffic_pen + HAZARD_W * haz)
        keys = list(edges.index)
        nx.set_edge_attributes(G, dict(zip(keys, cost)), "safe_cost")
        nx.set_edge_attributes(G, dict(zip(keys, base * activity)), "traffic")
        nx.set_edge_attributes(G, dict(zip(keys, edges["light"].to_numpy())), "light")
        nx.set_edge_attributes(G, dict(zip(keys, haz)), "haz")

        o, o_dist = ox.distance.nearest_nodes(G, X=from_lng, Y=from_lat, return_dist=True)
        d, d_dist = ox.distance.nearest_nodes(G, X=to_lng, Y=to_lat, return_dist=True)
        if o_dist > 400:
            raise HTTPException(400, "The start point is outside the walking area.")
        if d_dist > 400:
            raise HTTPException(400, "The destination is outside the walking area.")
        try:
            fastest = nx.shortest_path(G, o, d, weight="length")
            safest = nx.shortest_path(G, o, d, weight="safe_cost")
        except nx.NetworkXNoPath:
            raise HTTPException(404, "No walking route found between those points.")
        return {"fastest": _summarize(G, fastest, "length"), "safest": _summarize(G, safest, "safe_cost")}


# Mount LAST so /api routes take priority
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)