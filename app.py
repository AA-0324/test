import streamlit as st

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
    page_title="CampusWay",
    page_icon="🧭",
)

import time
import html
import re
import threading
import requests
from collections import deque
import pandas as pd
import geopandas as gpd
import folium
from branca.element import MacroElement, Template
from shapely.geometry import shape, box as shapelyBox
from folium import plugins as folium_plugins
import leafmap.foliumap as leafmap
import osmnx as ox

BUILD_MARKER = "diag-2026-09-12-05-latlonfallback"


MAX_FEATURES_PER_LAYER = 6000

# ── CARTO basemap tile URL with API key (removes watermark) ──────────────────
CARTO_KEY = "cb1_2ely_1_56403dce0becb94f8ac75d76"
CARTO_TILE_URL = (
    f"https://{{s}}.basemaps.cartocdn.com/rastertiles/light_all/{{z}}/{{x}}/{{y}}.png"
    f"?key={CARTO_KEY}"
)
CARTO_ATTRIBUTION = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, '
    '&copy; <a href="https://carto.com/attributions">CARTO</a>'
)


def _capFeatures(gdf, maxRows=MAX_FEATURES_PER_LAYER):
    if gdf is None or len(gdf) <= maxRows:
        return gdf
    try:
        areas = gdf.geometry.area
        return gdf.loc[areas.sort_values(ascending=False).index[:maxRows]]
    except Exception:
        return gdf.iloc[:maxRows]


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

WALKWAY_VALUES = ["footway", "path", "pedestrian", "steps"]
ROAD_VALUES = ["service", "residential", "unclassified", "living_street"]
FACILITY_AMENITY_VALUES = ["library", "food_court", "cafe"]
FACILITY_LEISURE_VALUES = ["sports_centre", "fitness_centre"]

layer_labels = {
    "buildings":  "Campus Buildings",
    "walkways":   "Pedestrian Paths",
    "roads":      "Roads & Service Routes",
    "facilities": "Specialized Facilities"
}

def styleFor(color, fill, opacity, weight, dashed=False):
    s = {"color": color, "fillColor": fill, "fillOpacity": opacity, "weight": weight, "opacity": 0.9}
    if dashed:
        s["dashArray"] = "6, 6"
    return lambda feat: s

campusStyles = {
    "buildings":  styleFor("#3A6EA5", "#3A6EA5", 0.45, 1.0),
    "walkways":   styleFor("#1F6F54", "#1F6F54", 0.0,  2.5, dashed=True),
    "roads":      styleFor("#726B5E", "#726B5E", 0.0,  1.5),
    "facilities": styleFor("#A02B5C", "#A02B5C", 0.7,  1.5),
}

LAYER_COLORS = {
    "buildings":  "#3A6EA5",
    "walkways":   "#1F6F54",
    "roads":      "#726B5E",
    "facilities": "#A02B5C",
}

req_headers = {"User-Agent": "global-campus-navigator/1.0 (streamlit-app)"}
RATE_LIMIT_GAP = 1.5


def runWithDeadline(fn, args, seconds, label):
    """
    Run fn(*args) in a background thread and give up after `seconds`,
    regardless of what fn is actually doing (blocked socket read, DNS
    resolution hang, a dependency ignoring its own timeout, etc).

    IMPORTANT: this deliberately does NOT use ThreadPoolExecutor as a
    `with` block. `with ThreadPoolExecutor() as pool:` calls
    pool.shutdown(wait=True) on exit -- which BLOCKS until the background
    thread actually finishes, even if future.result(timeout=...) already
    raised. That silently defeated the entire point of a deadline: the
    caller would still hang for however long the real call took, just
    with an extra exception swallowed at the end. Calling shutdown(wait=False)
    explicitly lets the calling thread return immediately once the deadline
    hits; the orphaned worker thread finishes on its own and its result
    (or exception) is simply discarded.
    """
    import concurrent.futures
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, *args)
    try:
        result = future.result(timeout=seconds)
        pool.shutdown(wait=False)
        return result
    except concurrent.futures.TimeoutError:
        pool.shutdown(wait=False)  # do NOT wait -- this is the actual fix
        raise TimeoutError(
            f"{label} took longer than {seconds}s and was cancelled. "
            f"OpenStreetMap's free services can be slow or unreachable "
            f"depending on network conditions -- try again in a moment."
        )
    except Exception:
        pool.shutdown(wait=False)
        raise


OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
]


@st.cache_resource
def initOsmnx():
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.requests_timeout = 25
    ox.settings.overpass_url = OVERPASS_MIRRORS[0]
    return True


def fetchFromOverpass(fetchFn, *args):
    # No _status param: called from inside st.cache_data functions where
    # writing to Streamlit widgets is forbidden and causes the cached-replay crash.
    # Try each mirror ONCE -- no retry-with-sleep here. Retrying each mirror
    # multiple times with long timeouts previously multiplied worst-case wait
    # to several minutes per call, which is indistinguishable from "stuck" to
    # a user. The hard deadline in runWithDeadline() is the real safety net.
    lastErr = None
    for mirror in OVERPASS_MIRRORS:
        ox.settings.overpass_url = mirror
        try:
            return fetchFn(*args)
        except Exception as e:
            lastErr = e
            continue
    ox.settings.overpass_url = OVERPASS_MIRRORS[0]
    raise RuntimeError(
        f"OpenStreetMap's Overpass data service didn't respond after trying "
        f"{len(OVERPASS_MIRRORS)} server(s). This is a shared free service and "
        f"it does get overloaded, especially for large campuses -- it's not a "
        f"sign that this campus lacks data. Raw error: {lastErr}"
    )


_nomLastCallTime = [0.0]
_nomLock = threading.Lock()

def throttleNominatim():
    # Plain module-level state, NOT st.session_state: this function is called
    # from inside a background thread (see runWithDeadline), and st.session_state
    # requires Streamlit's script-run context which only exists on the main
    # thread. Using session_state here was silently unreliable across threads.
    with _nomLock:
        gap = RATE_LIMIT_GAP - (time.monotonic() - _nomLastCallTime[0])
        if gap > 0:
            time.sleep(gap)
        _nomLastCallTime[0] = time.monotonic()


initOsmnx()

# ── campus detection ─────────────────────────────────────────────────────────
# NOT_CAMPUS previously included "boundary" -- but university campuses are
# frequently mapped in OSM as boundary relations (type=boundary,
# boundary=administrative or boundary=place), so that exclusion was silently
# rejecting real campuses like MIT, Caltech, and many others that happen to
# be stored as relation boundaries rather than amenity polygons in OSM.
# Removed "boundary" from the hard-exclusion set; the positive EDU_TAG_PAIRS
# and campusNameHints checks already handle the real filtering correctly.
NOT_CAMPUS = {"leisure", "shop", "tourism", "highway", "natural"}

EDU_TAG_PAIRS = {
    ("amenity", "university"),
    ("amenity", "college"),
    ("amenity", "school"),
    ("amenity", "research_institute"),
    ("landuse", "education"),
    ("boundary", "educational"),    # additional OSM tagging pattern used for
    ("place", "campus"),            # many large US/UK/AU research campuses
}

campusNameHints = (
    "university",
    "college",
    "institute of technology",
    "polytechnic",
    "academy",
    "ecole",
    "universidad",
    "universita",
    "mit",          # well-known acronyms that don't contain a hint word
    "caltech",
    "mit.edu",
)


def looksLikeCampus(nominatimResult):
    pair = (nominatimResult.get("class"), nominatimResult.get("type"))
    if pair in EDU_TAG_PAIRS:
        return True
    # Hard-exclusion: classes that are unambiguously NOT a campus no matter
    # what their display name says (a shop named "University Bookstore" should
    # not be treated as a campus result).
    if nominatimResult.get("class") in NOT_CAMPUS:
        return False
    dn = nominatimResult.get("display_name") or ""
    if not isinstance(dn, str):
        dn = str(dn)
    return any(hint in dn.lower() for hint in campusNameHints)


PHOTON_URL = "https://photon.komoot.io/api/"


def queryPhoton(q, limit=5):
    p = {"q": q, "limit": limit}
    r = requests.get(PHOTON_URL, params=p, headers=req_headers, timeout=10)
    r.raise_for_status()
    data = r.json()

    out = []
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        name = props.get("name") or q
        parts = [name, props.get("city"), props.get("state"), props.get("country")]
        display_name = ", ".join(x for x in parts if x)

        boundingbox = None
        extent = props.get("extent")
        if extent and len(extent) == 4:
            minLon, maxLat, maxLon, minLat = extent
            boundingbox = [str(minLat), str(maxLat), str(minLon), str(maxLon)]

        out.append({
            "class": props.get("osm_key"),
            "type": props.get("osm_value"),
            "display_name": display_name,
            "osm_type": props.get("osm_type"),
            "osm_id": props.get("osm_id"),
            "geojson": None,
            "boundingbox": boundingbox,
            "lat": coords[1] if coords else None,
            "lon": coords[0] if coords else None,
        })
    return out


def queryNominatim(q, limit=5):
    p = {
        "q": q,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
        "polygon_geojson": 1,
    }
    maxAttempts = 2
    backoffs = [2, 5]
    lastStatus = None
    lastBody = None

    for attempt in range(maxAttempts):
        throttleNominatim()
        try:
            r = requests.get(NOMINATIM_URL, params=p, headers=req_headers, timeout=10)
        except requests.exceptions.RequestException as e:
            if attempt == maxAttempts - 1:
                raise ValueError(f"Could not reach OpenStreetMap's search service: `{e}`")
            time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
            continue

        if r.status_code != 429:
            r.raise_for_status()
            return r.json()

        lastStatus = r.status_code
        lastBody = (r.text or "")[:300]
        if attempt == maxAttempts - 1:
            break

        retryAfter = r.headers.get("Retry-After")
        try:
            wait = float(retryAfter) if retryAfter is not None else backoffs[min(attempt, len(backoffs) - 1)]
        except ValueError:
            wait = backoffs[min(attempt, len(backoffs) - 1)]
        wait = min(wait, 10)

        time.sleep(wait)

    raise ValueError(
        f"OpenStreetMap's search returned **HTTP {lastStatus}** on every attempt just now.\n\n"
        f"Raw response: `{lastBody or '(empty)'}`"
    )


FALLBACK_TAG = {
    "buildings":  ("building", "Building"),
    "facilities": ("amenity", "Facility"),
    "walkways":   ("highway", "Path"),
    "roads":      ("highway", "Road"),
}


USELESS_CATEGORY_VALUES = {"yes", "university", "college"}


def _friendlySeries(s):
    is_str = s.map(lambda v: isinstance(v, str))
    s = s.where(is_str)
    s = s.str.strip()
    useless = s.str.lower().isin(USELESS_CATEGORY_VALUES)
    s = s.mask(useless | (s == ""))
    return s.str.replace("_", " ", regex=False).str.replace("-", " ", regex=False).str.title()


def _friendly(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() in USELESS_CATEGORY_VALUES:
        return None
    return value.replace("_", " ").replace("-", " ").title()


def addLabelAndTrim(gdf, layer_key):
    if gdf is None or gdf.empty:
        return gdf
    geom_col = gdf.geometry.name
    keep_cols = [geom_col]
    if "osmid" in gdf.columns:
        keep_cols.append("osmid")
    try:
        gdf = gdf.copy()
        tag_col, fallback_word = FALLBACK_TAG[layer_key]

        name_s = gdf["name"] if "name" in gdf.columns else pd.Series([None] * len(gdf), index=gdf.index)
        name_s = name_s.map(lambda v: v.strip() if isinstance(v, str) else None)
        has_name = name_s.notna() & (name_s != "")

        primary_col = gdf[tag_col] if tag_col in gdf.columns else pd.Series([None] * len(gdf), index=gdf.index)
        category = _friendlySeries(primary_col)

        if layer_key == "facilities" and "leisure" in gdf.columns:
            category = category.fillna(_friendlySeries(gdf["leisure"]))

        label = name_s.where(has_name, category.fillna(fallback_word))

        gdf["Label"] = label
        gdf["HasName"] = has_name
        gdf["Category"] = category
        keep_cols += ["Label", "HasName", "Category"]
        return gdf[keep_cols]
    except Exception:
        return gdf


SIMPLIFY_TOLERANCE = 0.00005


def _simplify(gdf):
    if gdf is None or gdf.empty:
        return gdf
    try:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
        return gdf
    except Exception:
        return gdf


def makeTooltip():
    return folium.GeoJsonTooltip(fields=["Label"], labels=False, sticky=False)


def _mergeBounds(existing, south, west, north, east):
    if existing is None:
        return (south, west, north, east)
    es, ew, en, ee = existing
    return (min(es, south), min(ew, west), max(en, north), max(ee, east))


def _haversineMeters(lat1, lon1, lat2, lon2):
    import math
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def nearbyLocations(focusLoc, focusName, allLocs, maxResults=4, maxMeters=300):
    if not focusLoc:
        return []
    flat, flon = focusLoc
    out = []
    for name, (lat, lon) in allLocs.items():
        if name == focusName:
            continue
        d = _haversineMeters(flat, flon, lat, lon)
        if d <= maxMeters:
            out.append((name, d))
    out.sort(key=lambda x: x[1])
    return out[:maxResults]


def _editDistance(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def _fuzzyScore(query, candidate):
    q = query.lower().strip()
    c = candidate.lower()
    if not q:
        return 0
    if q in c:
        return 1000 - c.index(q) - abs(len(c) - len(q)) * 0.1
    words = re.split(r"[\s()]+", c)
    candidates = [w for w in words if w] + [c]
    bestWord, best = min(((w, _editDistance(q, w)) for w in candidates), key=lambda t: t[1])
    maxAllowed = max(1, min(len(q), len(bestWord)) // 2)
    if best > maxAllowed:
        return 0
    return 500 - best * 40


def fuzzySearch(query, names, limit=6, minScore=1):
    scored = [(n, _fuzzyScore(query, n)) for n in names]
    scored = [t for t in scored if t[1] >= minScore]
    scored.sort(key=lambda t: -t[1])
    return [n for n, _ in scored[:limit]]


def _geomBounds(geometry):
    if not geometry:
        return None
    pts = []

    def _walk(c):
        if not isinstance(c, (list, tuple)) or not c:
            return
        if isinstance(c[0], (int, float)):
            pts.append(c)
        else:
            for sub in c:
                _walk(sub)

    _walk(geometry.get("coordinates"))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _roundGeoJson(geo, precision=5):
    if not geo:
        return geo

    def _round(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            return [round(v, precision) for v in c]
        if isinstance(c, (list, tuple)):
            return [_round(x) for x in c]
        return c

    for feat in geo.get("features", []):
        geom = feat.get("geometry")
        if geom and geom.get("coordinates") is not None:
            geom["coordinates"] = _round(geom["coordinates"])
    return geo


def _stripUnusedProps(geo, keep=("Label",)):
    if not geo:
        return geo
    for feat in geo.get("features", []):
        props = feat.get("properties")
        if props:
            feat["properties"] = {k: v for k, v in props.items() if k in keep}
    return geo


def _segmentEndpoints(geometry):
    if not geometry:
        return []
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")
    if not coords:
        return []
    pts = []
    if gtype == "LineString" and coords:
        pts.append(tuple(coords[0][:2]))
        pts.append(tuple(coords[-1][:2]))
    elif gtype == "MultiLineString":
        for line in coords:
            if line:
                pts.append(tuple(line[0][:2]))
                pts.append(tuple(line[-1][:2]))
    return pts


def _propagateRoadNames(features, maxHops=6):
    endpointOwners = {}
    for i, feat in enumerate(features):
        for pt in _segmentEndpoints(feat.get("geometry")):
            endpointOwners.setdefault(pt, []).append(i)

    effectiveName = {}
    queue = deque()
    for i, feat in enumerate(features):
        props = feat.get("properties") or {}
        if props.get("HasName"):
            name = (props.get("Label") or "").strip()
            if name:
                effectiveName[i] = name
                queue.append((i, 0))

    while queue:
        i, hops = queue.popleft()
        if hops >= maxHops:
            continue
        name = effectiveName[i]
        for pt in _segmentEndpoints(features[i].get("geometry")):
            for j in endpointOwners.get(pt, ()):
                if j in effectiveName:
                    continue
                effectiveName[j] = name
                queue.append((j, hops + 1))

    return effectiveName


def fetchRoadsAndWalkways(polygon_wkt):
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)

    # osmnx DOES support list values as OR filters (confirmed against docs) --
    # scope this to only the highway types we display, rather than True
    # (which pulls every highway type, including motorways/trunks/etc that
    # get filtered right back out below -- unnecessarily slow for no benefit).
    allowed = WALKWAY_VALUES + ROAD_VALUES
    gdf = fetchFromOverpass(ox.features_from_polygon, poly, {"highway": allowed})
    if gdf is None or gdf.empty or "highway" not in gdf.columns:
        return None, None, {}, {}
    gdf = gdf.to_crs("EPSG:4326") if gdf.crs else gdf
    gdf = _simplify(gdf)
    gdf = _capFeatures(gdf)

    walkways = gdf[gdf["highway"].isin(WALKWAY_VALUES)].copy()
    roads = gdf[gdf["highway"].isin(ROAD_VALUES)].copy()

    walkways = addLabelAndTrim(walkways, "walkways")
    roads = addLabelAndTrim(roads, "roads")
    walkGeo = walkways.__geo_interface__ if walkways is not None and not walkways.empty else None
    roadGeo = roads.__geo_interface__ if roads is not None and not roads.empty else None
    walkGeo = _roundGeoJson(walkGeo)
    roadGeo = _roundGeoJson(roadGeo)

    allFeatures = []
    for geo in (roadGeo, walkGeo):
        if geo:
            allFeatures.extend(geo.get("features", []))

    effectiveNames = _propagateRoadNames(allFeatures)

    namedRoads = {}
    namedRoadGeo = {}
    for i, feat in enumerate(allFeatures):
        name = effectiveNames.get(i)
        if not name:
            continue

        b = _geomBounds(feat.get("geometry"))
        if b:
            minx, miny, maxx, maxy = b
            namedRoads[name] = _mergeBounds(namedRoads.get(name), miny, minx, maxy, maxx)

        if name not in namedRoadGeo:
            namedRoadGeo[name] = {"type": "FeatureCollection", "features": [feat]}
        else:
            namedRoadGeo[name]["features"].append(feat)

    roadGeo = _stripUnusedProps(roadGeo)
    walkGeo = _stripUnusedProps(walkGeo)

    return roadGeo, walkGeo, namedRoads, namedRoadGeo
fetchRoadsAndWalkways = st.cache_data(show_spinner=False, ttl="24h")(fetchRoadsAndWalkways)


def fetchBuildingsAndFacilities(polygon_wkt):
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)

    # osmnx DOES support list values as OR filters (confirmed against docs) --
    # scope amenity/leisure to only the values we actually display, rather than
    # True (which pulls down every amenity of any kind -- benches, waste
    # baskets, vending machines, etc -- across the whole campus and is much
    # slower for large campuses like MIT for no benefit, since we filter it
    # right back down below anyway).
    gdf = fetchFromOverpass(ox.features_from_polygon, poly, {
        "building": True,
        "amenity": FACILITY_AMENITY_VALUES,
        "leisure": FACILITY_LEISURE_VALUES,
    })
    if gdf is None or gdf.empty:
        return None, None
    gdf = gdf.to_crs("EPSG:4326") if gdf.crs else gdf
    gdf = _simplify(gdf)
    gdf = _capFeatures(gdf)

    has_building = gdf["building"].notna() if "building" in gdf.columns else pd.Series(False, index=gdf.index)
    has_amenity = gdf["amenity"].isin(FACILITY_AMENITY_VALUES) if "amenity" in gdf.columns else pd.Series(False, index=gdf.index)
    has_leisure = gdf["leisure"].isin(FACILITY_LEISURE_VALUES) if "leisure" in gdf.columns else pd.Series(False, index=gdf.index)

    buildings = gdf[has_building].copy()
    facilities = gdf[has_amenity | has_leisure].copy()

    buildings = addLabelAndTrim(buildings, "buildings")
    facilities = addLabelAndTrim(facilities, "facilities")
    return buildings, facilities
fetchBuildingsAndFacilities = st.cache_data(show_spinner=False, ttl="24h")(fetchBuildingsAndFacilities)


def stripDuplicateBuildings(buildingsDf, facilitiesDf):
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


def _alternateNameGuess(name):
    m = re.match(r'^(university|college)\s+of\s+(.+)$', name.strip(), re.IGNORECASE)
    if not m:
        return None
    kind, rest = m.groups()
    return f"{rest.strip()} {kind.title()}"


def findCampus(name):
    try:
        results = queryNominatim(name)
    except ValueError as nomErr:
        try:
            results = queryPhoton(name)
        except Exception:
            raise nomErr

    if not results:
        raise ValueError(f'No results found for **"{name}"** on OpenStreetMap.\n\nTry a more specific name, e.g. `"{name}, City, Country"`')

    results.sort(key=lambda r: not looksLikeCampus(r))
    edu_hits = [r for r in results if looksLikeCampus(r)]

    if not edu_hits:
        alt = _alternateNameGuess(name)
        if alt:
            try:
                altResults = queryNominatim(alt)
                altResults.sort(key=lambda r: not looksLikeCampus(r))
                altEduHits = [r for r in altResults if looksLikeCampus(r)]
                if altEduHits:
                    edu_hits = altEduHits
                    results = altResults
            except Exception:
                pass

    if not edu_hits:
        # Last-ditch: if the query itself contains an education keyword but
        # none of the Nominatim results matched our classifier, accept the
        # top result anyway rather than showing a hard failure -- the user
        # knows what they searched for.
        name_lower = name.lower()
        query_looks_educational = any(hint in name_lower for hint in campusNameHints)
        if query_looks_educational and results:
            edu_hits = results[:1]

    if not edu_hits:
        raise ValueError(
            f'None of the search results for **"{name}"** look like a university or college.\n\n'
            f'Try the institution\'s full official name (for example, "Carleton University" '
            f'rather than "University of Carleton"), or add a city/country for a more specific match.'
        )

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

    # No hit had a usable polygon. Before falling back to a raw bounding box,
    # prefer a "relation" or "way" hit over a bare "node" (point) -- a node's
    # Nominatim bbox is tiny (tens of meters), which for a real campus almost
    # always misses every actual building/road, silently producing "no data"
    # even though OSM has plenty of data for the school. Prefer whichever hit
    # has the largest bbox area as a proxy for "most likely a real boundary".
    def _bboxArea(hit):
        bbox = hit.get("boundingbox")
        if not bbox or len(bbox) != 4:
            return 0.0
        try:
            south, north, west, east = (float(v) for v in bbox)
            return max(0.0, north - south) * max(0.0, east - west)
        except Exception:
            return 0.0

    typeRank = {"relation": 0, "way": 1, "node": 2}
    ranked = sorted(
        edu_hits,
        key=lambda h: (typeRank.get(h.get("osm_type"), 3), -_bboxArea(h))
    )
    top_hit = ranked[0]
    hitName = top_hit.get("display_name", name)

    MIN_SPAN_DEG = 0.0018  # ~200m at mid latitudes, so ~400m box

    bbox = top_hit.get("boundingbox")
    if bbox and len(bbox) == 4:
        try:
            south, north, west, east = (float(v) for v in bbox)
            # A bare node's bbox is a tiny square around one point -- pad it
            # out to a sane minimum footprint (roughly 400m x 400m) so the
            # Overpass query has a real chance of catching nearby campus
            # buildings/roads instead of guaranteed-empty results.
            if (north - south) < MIN_SPAN_DEG or (east - west) < MIN_SPAN_DEG:
                cy, cx = (north + south) / 2, (east + west) / 2
                south, north = cy - MIN_SPAN_DEG, cy + MIN_SPAN_DEG
                west, east = cx - MIN_SPAN_DEG, cx + MIN_SPAN_DEG
            g = shapelyBox(west, south, east, north)
            return hitName, g.wkt
        except Exception:
            pass

    # No boundingbox at all -- this is the common case for Photon-sourced
    # results (queryPhoton only sets boundingbox when the source feature has
    # an "extent", which point-type results usually lack). Photon and
    # Nominatim both always provide lat/lon though, so build a padded box
    # around the point instead of giving up entirely.
    lat, lon = top_hit.get("lat"), top_hit.get("lon")
    if lat is not None and lon is not None:
        try:
            lat, lon = float(lat), float(lon)
            g = shapelyBox(lon - MIN_SPAN_DEG, lat - MIN_SPAN_DEG,
                            lon + MIN_SPAN_DEG, lat + MIN_SPAN_DEG)
            return hitName, g.wkt
        except Exception:
            pass

    raise ValueError(
        f'**"{hitName}"** was found but has no boundary polygon, bounding box, or even '
        f'coordinates on OpenStreetMap.\n\n'
        f'Try a more specific search or check [openstreetmap.org](https://www.openstreetmap.org).'
    )
findCampus = st.cache_data(show_spinner=False, ttl="24h")(findCampus)


def prepareCampusData(polygon_wkt, active_layers, status):
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)
    minx, miny, maxx, maxy = poly.bounds

    layerData = {}

    status.update(label="Fetching roads and pedestrian paths...")
    roadGeo, walkGeo, namedRoads, namedRoadGeo = None, None, {}, {}
    try:
        roadGeo, walkGeo, namedRoads, namedRoadGeo = runWithDeadline(
            fetchRoadsAndWalkways, (polygon_wkt,), 75, "Road/path fetch"
        )
        status.write(f"Roads: {len(roadGeo['features']) if roadGeo else 0} segments, "
                     f"Paths: {len(walkGeo['features']) if walkGeo else 0} segments")
    except Exception as e:
        status.write(f"⚠️ Road/path data unavailable ({e}) — continuing with buildings only.")
    layerData["roads"] = roadGeo
    layerData["walkways"] = walkGeo

    status.update(label="Fetching buildings and facilities...")
    bldGdf, facGdf = None, None
    try:
        bldGdf, facGdf = runWithDeadline(
            fetchBuildingsAndFacilities, (polygon_wkt,), 75, "Building fetch"
        )
    except Exception as e:
        status.write(f"⚠️ Building data unavailable ({e}) — continuing with roads/paths only.")

    bldGdf = stripDuplicateBuildings(bldGdf, facGdf)
    layerData["buildings"] = _stripUnusedProps(_roundGeoJson(bldGdf.__geo_interface__)) if bldGdf is not None and not bldGdf.empty else None
    layerData["facilities"] = _stripUnusedProps(_roundGeoJson(facGdf.__geo_interface__)) if facGdf is not None and not facGdf.empty else None
    status.write(f"Buildings: {len(bldGdf) if bldGdf is not None else 0}, "
                 f"Facilities: {len(facGdf) if facGdf is not None else 0}")

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
        foundAnything = True

    if not foundAnything:
        area_deg2 = (maxx - minx) * (maxy - miny)
        raise ValueError(
            f"OSM has no roads, paths, buildings, or facilities tagged inside the matched "
            f"boundary for this campus (searched area: roughly {minx:.4f},{miny:.4f} to "
            f"{maxx:.4f},{maxy:.4f}, ~{area_deg2*111*111:.2f} km²). This usually means the "
            f"name matched a point or a very small/wrong area on OpenStreetMap rather than "
            f"the actual campus footprint. Try adding the city and state, or check the name "
            f"on openstreetmap.org."
        )

    status.write("Indexing named buildings and roads for search...")

    namedLocations = {}
    for gdf in (bldGdf, facGdf):
        if gdf is None or gdf.empty or "HasName" not in gdf.columns:
            continue
        for _, row in gdf[gdf["HasName"] == True].iterrows():
            try:
                centroid = row.geometry.centroid
                label = row["Label"]
                category = row.get("Category")
                hasCategory = isinstance(category, str) and pd.notna(category) and category.strip()
                display = f"{label} ({category})" if hasCategory else label
                key = display
                n = 2
                while key in namedLocations:
                    key = f"{display} #{n}"
                    n += 1
                namedLocations[key] = (centroid.y, centroid.x)
            except Exception:
                continue

    bounds = (miny, minx, maxy, maxx)
    return layerData, counts, namedLocations, namedRoads, namedRoadGeo, bounds


def addBranding(m):
    branding_html = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; bottom: 10px; right: 10px; z-index: 9999;
                background: white; padding: 6px 12px; border-radius: 4px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 12px;
                font-family: -apple-system, sans-serif; font-weight: 600;
                color: #333;">
        Campus<span style="color:#C54C0A;">Way</span>
    </div>
    {% endmacro %}
    """
    branding = MacroElement()
    branding._template = Template(branding_html)
    m.get_root().add_child(branding)


def renderCampusMap(layerData, counts, bounds, visibleLayers, focusName=None, focusLoc=None,
                     focusRoad=None, focusRoadBounds=None, focusRoadGeo=None):
    miny, minx, maxy, maxx = bounds
    cLat = (miny + maxy) / 2
    cLon = (minx + maxx) / 2

    # Use explicit CARTO tile URL with API key instead of the named alias so
    # the key parameter is included and the "API key required" watermark is
    # removed. CartoDB.Positron (light_all) is visually identical to what the
    # app used before -- the only change is the key in the URL.
    m = leafmap.Map(
        center=[cLat, cLon],
        zoom=15,
        tiles=None,          # disable the default tile layer; we add ours below
        prefer_canvas=True,
        control_scale=True,
    )
    folium.TileLayer(
        tiles=CARTO_TILE_URL,
        attr=CARTO_ATTRIBUTION,
        name="CartoDB Positron",
        subdomains=["a", "b", "c", "d"],
        max_zoom=19,
    ).add_to(m)

    folium_plugins.Fullscreen(
        position="topright",
        title="Expand map",
        title_cancel="Exit fullscreen",
        force_separate_button=True
    ).add_to(m)

    folium_plugins.LocateControl(
        position="topright",
        strings={"title": "Show my location"},
        flyTo=True,
    ).add_to(m)

    folium_plugins.MeasureControl(
        position="topleft",
        primary_length_unit="meters",
        secondary_length_unit="feet",
    ).add_to(m)

    drawOrder = ["roads", "walkways", "buildings", "facilities"]
    for k in drawOrder:
        if k not in visibleLayers:
            continue
        geo = layerData.get(k)
        if not geo or not geo.get("features"):
            continue
        folium.GeoJson(
            data=geo,
            name=layer_labels[k],
            style_function=campusStyles[k],
            tooltip=makeTooltip(),
            smooth_factor=1.5,
        ).add_to(m)

    rendered = [k for k in drawOrder if k in visibleLayers and counts.get(k, 0) > 0]
    if rendered:
        rows = "".join(
            f'<div style="margin:2px 0;">'
            f'<span style="display:inline-block;width:12px;height:12px;background:{LAYER_COLORS[k]};'
            f'margin-right:6px;border-radius:2px;"></span>{layer_labels[k]}</div>'
            for k in rendered
        )
        legend_html = f"""
        {{% macro html(this, kwargs) %}}
        <div style="position: fixed; bottom: 30px; left: 10px; z-index: 9999;
                    background: #FFFFFF; padding: 8px 12px; border-radius: 3px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 12.5px;
                    font-family: -apple-system, sans-serif; line-height: 1.5;
                    border-top: 3px solid #16233B;">
            {rows}
        </div>
        {{% endmacro %}}
        """
        legend = MacroElement()
        legend._template = Template(legend_html)
        m.get_root().add_child(legend)

    addBranding(m)

    if focusName and focusLoc:
        flat, flon = focusLoc
        pad = 0.0012
        m.fit_bounds([[flat - pad, flon - pad], [flat + pad, flon + pad]])
        folium.CircleMarker(
            location=[flat, flon],
            radius=16,
            color="#C54C0A",
            weight=3,
            fill=True,
            fill_color="#C54C0A",
            fill_opacity=0.15,
            tooltip=folium.Tooltip(focusName),
        ).add_to(m)
    elif focusRoad and focusRoadBounds:
        s, w, n, e = focusRoadBounds
        padLat = max((n - s) * 0.15, 0.0005)
        padLon = max((e - w) * 0.15, 0.0005)
        m.fit_bounds([[s - padLat, w - padLon], [n + padLat, e + padLon]])
        if focusRoadGeo:
            folium.GeoJson(
                data=focusRoadGeo,
                name="__highlight__",
                style_function=lambda feat: {
                    "color": "#C54C0A",
                    "weight": 6,
                    "opacity": 0.9,
                },
                tooltip=folium.Tooltip(focusRoad),
            ).add_to(m)
    else:
        m.fit_bounds([[miny, minx], [maxy, maxx]])

    return m


# ── visual identity ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@700;800&family=IBM+Plex+Mono:wght@500&display=swap');

:root {
    --cw-ink: #16233B;
    --cw-accent: #C54C0A;
    --cw-accent-hover: #A23E08;
    --cw-muted: #5B6472;
}

.stButton button[kind="primary"] {
    background: var(--cw-accent) !important;
    border-color: var(--cw-accent) !important;
}
.stButton button[kind="primary"]:hover {
    background: var(--cw-accent-hover) !important;
    border-color: var(--cw-accent-hover) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex; align-items:center; gap:0.85rem;
            padding: 0.4rem 0 1.1rem 0; border-bottom: 2px solid #E5E5E0;
            margin-bottom: 1.1rem;">
    <div style="width:40px; height:40px; flex-shrink:0; background: var(--cw-ink);
                border-radius:6px; display:flex; align-items:center; justify-content:center;">
        <span style="font-size:1.25rem; line-height:1;">🧭</span>
    </div>
    <div>
        <div style="font-family:'Barlow Condensed', sans-serif; font-weight:800;
                    font-size:1.7rem; line-height:1; color: var(--cw-ink);">
            CAMPUS<span style="color: var(--cw-accent);">WAY</span>
        </div>
        <div style="font-family:'IBM Plex Mono', monospace; font-size:0.75rem;
                    color: var(--cw-muted); margin-top:3px;">
            Wayfinding for any campus, anywhere
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── sidebar ──────────────────────────────────────────────────────────────────

use_facilities = True

with st.sidebar:
    st.caption(f"build: {BUILD_MARKER}")

    with st.expander("🔧 Network diagnostic (isolated test)"):
        st.caption(
            "This bypasses ALL app logic and makes ONE direct request to "
            "Nominatim with a 10s timeout. If this hangs or errors, the "
            "problem is network/deployment-level, not this app's code."
        )
        if st.button("Run isolated network test"):
            diagStart = time.time()
            try:
                diagResp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": "MIT", "format": "jsonv2", "limit": 1},
                    headers={"User-Agent": "campusway-diagnostic/1.0"},
                    timeout=10,
                )
                diagElapsed = time.time() - diagStart
                st.success(f"HTTP {diagResp.status_code} in {diagElapsed:.1f}s")
                st.json(diagResp.json()[:1] if diagResp.ok else diagResp.text[:500])
            except Exception as e:
                diagElapsed = time.time() - diagStart
                st.error(f"FAILED after {diagElapsed:.1f}s: {type(e).__name__}: {e}")

    st.divider()
    st.subheader("Search")
    campusInput = st.text_input(
        "University or college name",
        placeholder='e.g. "MIT" or "Foothill College, CA"',
        help="Full names, partial names, and acronyms all work. Add a city or country if you get the wrong result."
    )
    searchBtn = st.button("Generate Map", type="primary", use_container_width=True)

    with st.expander("How to use this map"):
        st.markdown(
            "- Use **Quick search** below to jump to a building or road, even if you don't spell it exactly right.\n"
            "- Tap the location icon on the map (top right) to show where you are right now.\n"
            "- Use the ruler icon (top left) to measure a real walking distance.\n"
            "- Toggle layers below to show or hide what's on the map."
        )

    st.divider()
    st.subheader("Layers")
    col1, col2 = st.columns(2)
    with col1:
        showBuildings = st.checkbox("Buildings", value=True)
        showRoads     = st.checkbox("Roads",     value=True)
    with col2:
        showPaths      = st.checkbox("Paths",      value=True)
        showFacilities = st.checkbox("Facilities", value=use_facilities)

    active_layers = []
    if showPaths:      active_layers.append("walkways")
    if showBuildings:  active_layers.append("buildings")
    if showFacilities: active_layers.append("facilities")
    if showRoads:      active_layers.append("roads")

    namedLocs = st.session_state.get("namedLocations", {})
    namedRds = st.session_state.get("namedRoads", {})
    focusName = None
    focusRoad = None

    def _clearRoadSelection():
        st.session_state["road_select"] = None

    def _clearBuildingSelection():
        st.session_state["building_select"] = None

    if namedLocs or namedRds:
        st.divider()
        st.subheader("Navigate")

        quickQuery = st.text_input(
            "Quick search (handles typos)",
            key="quick_search",
            placeholder='e.g. "libary" or "quad"',
        )
        if quickQuery.strip():
            allNames = {n: "building" for n in namedLocs}
            allNames.update({n: "road" for n in namedRds})
            matches = fuzzySearch(quickQuery, list(allNames.keys()))
            if matches:
                for m in matches:
                    kind = allNames[m]
                    icon = "🏢" if kind == "building" else "🛣️"
                    if st.button(f"{icon} {m}", key=f"quick_pick_{kind}_{m}", use_container_width=True):
                        if kind == "building":
                            st.session_state["building_select"] = m
                            st.session_state["road_select"] = None
                        else:
                            st.session_state["road_select"] = m
                            st.session_state["building_select"] = None
            else:
                st.caption("No matches -- try a different spelling.")

        st.caption("Or browse the full list:")

        if namedLocs:
            selected = st.selectbox(
                "Find a building",
                sorted(namedLocs.keys()),
                index=None,
                placeholder="Type to search buildings...",
                key="building_select",
                on_change=_clearRoadSelection,
            )
            if selected:
                focusName = selected

        if namedRds:
            selectedRoad = st.selectbox(
                "Find a road",
                sorted(namedRds.keys()),
                index=None,
                placeholder="Type to search roads...",
                key="road_select",
                on_change=_clearBuildingSelection,
            )
            if selectedRoad:
                focusRoad = selectedRoad


# ── main area ─────────────────────────────────────────────────────────────────

if searchBtn and campusInput.strip():
    newSearch = campusInput.strip()
    if newSearch != st.session_state.get("lastSearch", ""):
        st.session_state["lastSearch"] = newSearch
        st.session_state.pop("campusData", None)
        st.session_state.pop("namedLocations", None)
        st.session_state.pop("namedRoads", None)
        st.session_state.pop("namedRoadGeo", None)

searchTerm = st.session_state.get("lastSearch", "")

if not searchTerm:
    st.info("Enter a university or college name in the sidebar and click **Generate Map** to get started.")
    st.stop()

if "campusData" not in st.session_state:
    err = None
    with st.status(f'Looking up "{searchTerm}"... (large campuses can take 30-60s)', expanded=True) as status:
        try:
            campusName, campusPoly = runWithDeadline(findCampus, (searchTerm,), 60, "Campus lookup")
            status.update(label=f"Found: {campusName}", state="running")
            status.write(f"Matched: {campusName}")
        except TimeoutError as e:
            status.update(label="Timed out", state="error")
            err = ("error", str(e))
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
                layerData, layerCounts, namedLocations, namedRoads, namedRoadGeo, bounds = \
                    prepareCampusData(campusPoly, active_layers, status)
                st.session_state["campusData"] = {
                    "layerData": layerData,
                    "counts": layerCounts,
                    "bounds": bounds,
                    "campusName": campusName,
                }
                st.session_state["namedLocations"] = namedLocations
                st.session_state["namedRoads"] = namedRoads
                st.session_state["namedRoadGeo"] = namedRoadGeo
                status.update(label=f"Map ready - {campusName}", state="complete", expanded=False)
            except ValueError as e:
                status.update(label="Couldn't build the map", state="error")
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

    st.rerun()

campusData = st.session_state["campusData"]

focusLoc = None
if focusName and focusName in st.session_state.get("namedLocations", {}):
    focusLoc = st.session_state["namedLocations"][focusName]
else:
    focusName = None

focusRoadBounds = None
focusRoadGeo = None
if focusRoad and focusRoad in st.session_state.get("namedRoads", {}):
    focusRoadBounds = st.session_state["namedRoads"][focusRoad]
    focusRoadGeo = st.session_state.get("namedRoadGeo", {}).get(focusRoad)
else:
    focusRoad = None

campusMap = renderCampusMap(
    campusData["layerData"],
    campusData["counts"],
    campusData["bounds"],
    visibleLayers=set(active_layers),
    focusName=focusName,
    focusLoc=focusLoc,
    focusRoad=focusRoad,
    focusRoadBounds=focusRoadBounds,
    focusRoadGeo=focusRoadGeo,
)

campusMap.to_streamlit(height=620)

if focusName and focusLoc:
    nearby = nearbyLocations(focusLoc, focusName, st.session_state.get("namedLocations", {}))
    if nearby:
        chips = " · ".join(f"{n} ({int(round(d))}m)" for n, d in nearby)
        st.caption(f"🧭 Near **{focusName}**: {chips}")
