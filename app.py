import streamlit as st

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

import time
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from folium import plugins as folium_plugins
import leafmap.foliumap as leafmap
import osmnx as ox


def getIds(gdf):
    if gdf is None or gdf.empty:
        return set()
    try:
        if isinstance(gdf.index, pd.MultiIndex):
            if "osmid" in (gdf.index.names or []):
                return set(gdf.index.get_level_values("osmid"))
            return set(x[-1] if isinstance(x, tuple) else x for x in gdf.index)
        if "osmid" in gdf.columns:
            return set(gdf["osmid"])
    except Exception:
        pass
    return set()


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

campus_tags = {
    "buildings": {"building": True},
    "walkways": {"highway": ["footway", "path", "pedestrian", "steps"]},
    "roads": {"highway": ["service", "residential", "unclassified", "living_street"]},
    "facilities": {"amenity": ["library", "food_court", "cafe"], "leisure": ["sports_centre", "fitness_centre"]}
}

layer_labels = {
    "buildings": "Campus Buildings",
    "walkways": "Pedestrian Paths",
    "roads": "Roads & Service Routes",
    "facilities": "Specialized Facilities"
}

def styleFor(color, fill, opacity, weight):
    s = {"color": color, "fillColor": fill, "fillOpacity": opacity, "weight": weight, "opacity": 0.9}
    return lambda feat: s

campusStyles = {
    "buildings": styleFor("#1f77b4", "#1f77b4", 0.45, 1.0),
    "walkways": styleFor("#2ca02c", "#2ca02c", 0.0, 2.0),
    "roads": styleFor("#7f7f7f", "#7f7f7f", 0.0, 1.5),
    "facilities": styleFor("#d62728", "#d62728", 0.7, 1.5),
}

req_headers = {"User-Agent": "global-campus-navigator/1.0 (streamlit-app)"}
RATE_LIMIT_GAP = 1.1


@st.cache_resource
def initOsmnx():
    ox.settings.use_cache = True
    ox.settings.log_console = False
    # don't set overpass_settings manually -- it has {timeout} and {maxsize} placeholders
    # that osmnx fills in itself. if you hardcode it those tokens never get substituted
    # and your queries die after 30s with no useful error message. ask me how I know lol
    return True


def throttleNominatim():
    # session_state survives across reruns, a plain global doesn't -- learned this the
    # annoying way when the rate limiter stopped working and nominatim started rejecting stuff
    t_last = st.session_state.get("nom_last", 0.0)
    gap = RATE_LIMIT_GAP - (time.monotonic() - t_last)
    if gap > 0:
        time.sleep(gap)
    st.session_state["nom_last"] = time.monotonic()


initOsmnx()

NOT_CAMPUS = {"leisure", "shop", "tourism", "highway", "natural", "boundary"}

EDU_TAG_PAIRS = {
    ("amenity", "university"),
    ("amenity", "college"),
    ("amenity", "school"),
    ("amenity", "research_institute"),
    ("landuse", "education")
}

campusNameHints = ("university", "college", "institute of technology", "polytechnic", "academy", "ecole", "universidad", "universita")


def looksLikeCampus(nominatimResult):
    pair = (nominatimResult.get("class"), nominatimResult.get("type"))
    if pair in EDU_TAG_PAIRS:
        return True
    if nominatimResult.get("class") in NOT_CAMPUS:
        return False
    dn = nominatimResult.get("display_name") or ""
    if not isinstance(dn, str):
        dn = str(dn)
    return any(hint in dn.lower() for hint in campusNameHints)


def queryNominatim(q, limit=5):
    p = {
        "q": q,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
        "extratags": 1,
        "polygon_geojson": 1,
    }
    throttleNominatim()
    r = requests.get(NOMINATIM_URL, params=p, headers=req_headers, timeout=10)
    r.raise_for_status()
    return r.json()


def fetchOsmLayer(polygon_wkt, layer_key):
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)
    try:
        gdf = ox.features_from_polygon(poly, tags=campus_tags[layer_key])
        if gdf is None or gdf.empty:
            return None
        gdf = gdf.to_crs("EPSG:4326") if gdf.crs else gdf
        return gdf.__geo_interface__
    except Exception:
        return None
fetchOsmLayer = st.cache_data(show_spinner=False, ttl="6h")(fetchOsmLayer)


def fetchOsmLayerRaw(polygon_wkt, layer_key):
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)
    try:
        gdf = ox.features_from_polygon(poly, tags=campus_tags[layer_key])
        if gdf is None or gdf.empty:
            return None
        return gdf.to_crs("EPSG:4326") if gdf.crs else gdf
    except Exception:
        return None
fetchOsmLayerRaw = st.cache_data(show_spinner=False, ttl="6h")(fetchOsmLayerRaw)


def stripDuplicateBuildings(buildingsDf, facilitiesDf):
    # osm tags a lot of campus buildings as BOTH building=yes AND amenity=library
    # (or cafe, gym, etc), which means without this step the same polygon ends up
    # in both the buildings layer and the facilities layer, drawn on top of each other
    #
    # has to run BEFORE converting to geojson -- geopandas replaces the osm ids with
    # sequential integers in the geojson output, so you can't match them up afterwards
    if buildingsDf is None or buildingsDf.empty:
        return buildingsDf

    fac_ids = getIds(facilitiesDf)
    if not fac_ids:
        return buildingsDf

    try:
        bld_ids = getIds(buildingsDf)
        overlap = bld_ids & fac_ids
        if not overlap:
            return buildingsDf

        if isinstance(buildingsDf.index, pd.MultiIndex):
            idList = [x[-1] if isinstance(x, tuple) else x for x in buildingsDf.index]
        elif "osmid" in buildingsDf.columns:
            idList = list(buildingsDf["osmid"])
        else:
            return buildingsDf

        keep = [i not in overlap for i in idList]
        return buildingsDf[keep]
    except Exception:
        return buildingsDf


def findCampus(name):
    results = queryNominatim(name)
    if not results:
        raise ValueError(f'No results found for **"{name}"** on OpenStreetMap.\n\nTry a more specific name, e.g. `"{name}, City, Country"`')

    results.sort(key=lambda r: not looksLikeCampus(r))
    edu_hits = [r for r in results if looksLikeCampus(r)]

    for hit in edu_hits:
        geo = hit.get("geojson") or {}
        if geo.get("type") not in ("Polygon", "MultiPolygon"):
            continue
        try:
            g = shape(geo)
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty:
                continue
            return hit.get("display_name", name), g.wkt
        except Exception:
            continue

    top_hit = edu_hits[0] if edu_hits else results[0]
    hitName = top_hit.get("display_name", name)

    try:
        throttleNominatim()
        gdf = ox.geocode_to_gdf(hitName)
    except Exception as e:
        raise ValueError(f'Found **"{hitName}"** but could not get its boundary.\n\nError: `{e}`')

    if gdf.empty or gdf.iloc[0].geometry.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            f'**"{hitName}"** is in OSM but only as a point, not a boundary polygon.\n\n'
            f'Try a more specific search or check [openstreetmap.org](https://www.openstreetmap.org).'
        )

    g = gdf.iloc[0].geometry
    if not g.is_valid:
        g = g.buffer(0)
    return hitName, g.wkt
findCampus = st.cache_data(show_spinner=False, ttl="24h")(findCampus)


def buildCampusMap(polygon_wkt, active_layers, status):
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)
    minx, miny, maxx, maxy = poly.bounds
    cLat = (miny + maxy) / 2
    cLon = (minx + maxx) / 2

    m = leafmap.Map(center=[cLat, cLon], zoom=15, tiles="CartoDB.Positron")

    folium_plugins.Fullscreen(
        position="topright",
        title="Expand map",
        title_cancel="Exit fullscreen",
        force_separate_button=True
    ).add_to(m)

    layerData = {}

    for k in ("roads", "walkways"):
        if k in active_layers:
            status.update(label=f"Fetching {layer_labels[k]}...")
            layerData[k] = fetchOsmLayer(polygon_wkt, k)

    facGdf = None
    if "facilities" in active_layers:
        status.update(label=f"Fetching {layer_labels['facilities']}...")
        facGdf = fetchOsmLayerRaw(polygon_wkt, "facilities")
        layerData["facilities"] = facGdf.__geo_interface__ if facGdf is not None else None

    if "buildings" in active_layers:
        status.update(label=f"Fetching {layer_labels['buildings']}...")
        bldGdf = fetchOsmLayerRaw(polygon_wkt, "buildings")
        bldGdf = stripDuplicateBuildings(bldGdf, facGdf)
        layerData["buildings"] = bldGdf.__geo_interface__ if bldGdf is not None else None

    drawOrder = ["roads", "walkways", "buildings", "facilities"]
    counts = {}
    foundAnything = False

    for k in drawOrder:
        if k not in layerData:
            continue
        geo = layerData[k]
        if not geo or not geo.get("features"):
            counts[k] = 0
            continue
        counts[k] = len(geo["features"])
        m.add_geojson(geo, layer_name=layer_labels[k], style_function=campusStyles[k])
        foundAnything = True

    if not foundAnything:
        raise ValueError("OSM has no tagged data for this campus. Try a different campus or check openstreetmap.org.")

    m.fit_bounds([[miny, minx], [maxy, maxx]])

    return m, counts


use_facilities = True

with st.sidebar:
    campusInput = st.text_input(
        "University or college name",
        placeholder='e.g. "MIT" or "Foothill College, CA"',
        help="Full names, partial names, and acronyms all work. Add a city or country if you get the wrong result. Press Enter or click Generate Map below."
    )

    showBuildings = st.checkbox("Campus Buildings", value=True)
    showPaths = st.checkbox("Pedestrian Paths", value=True)
    showRoads = st.checkbox("Roads & Service Routes", value=True)
    showFacilities = st.checkbox("Specialized Facilities", value=use_facilities)

    searchBtn = st.button("Generate Map", type="primary", use_container_width=True)

    active_layers = []
    if showPaths:
        active_layers.append("walkways")
    if showBuildings:
        active_layers.append("buildings")
    if showFacilities:
        active_layers.append("facilities")
    if showRoads:
        active_layers.append("roads")

if not searchBtn or not campusInput.strip():
    st.stop()

searchTerm = campusInput.strip()
err = None

with st.status(f'Looking up "{searchTerm}"... (large campuses can take 30-60s)', expanded=True) as status:
    try:
        campusName, campusPoly = findCampus(searchTerm)
        status.update(label=f"Found: {campusName}", state="running")
    except ValueError as e:
        status.update(label="Could not find campus", state="error")
        err = ("error", str(e))
    except Exception as e:
        status.update(label="Unexpected error", state="error")
        err = ("error", f"Something went wrong: `{e}`")

    if err is None and not active_layers:
        status.update(label="No layers selected", state="error")
        err = ("warning", "Select at least one layer in the sidebar.")

    if err is None:
        try:
            campusMap, layerCounts = buildCampusMap(campusPoly, active_layers, status)
            status.update(label=f"Map ready - {campusName}", state="complete", expanded=False)
        except ValueError as e:
            status.update(label="No OSM data found", state="error")
            err = ("warning", str(e))
        except Exception as e:
            status.update(label="Map build failed", state="error")
            err = ("error", f"Could not build map: `{e}`")

if err is not None:
    kind, msg = err
    if kind == "error":
        st.error(msg)
    else:
        st.warning(msg)
    st.stop()

campusMap.to_streamlit(height=620)
