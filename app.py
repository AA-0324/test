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


MAX_FEATURES_PER_LAYER = 6000


def _capFeatures(gdf, maxRows=MAX_FEATURES_PER_LAYER):
    # A hard ceiling on how many raw features get processed further. Without
    # this, an unusually large or imprecise campus boundary can pull back
    # tens of thousands of individually tagged features from Overpass --
    # processing all of them (simplifying geometry, labeling, deduping) is
    # what made the app look frozen for minutes with no feedback at all.
    # Keeping the largest features by area is a reasonable way to trim for a
    # wayfinding map: tiny sheds and slivers matter far less than the real
    # buildings and roads.
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


OVERPASS_MIRRORS = [
    "https://overpass-api.de/api",       # osmnx's default -- frequently overloaded
    "https://overpass.kumi.systems/api", # independent mirror, different infra/load
]


@st.cache_resource
def initOsmnx():
    ox.settings.use_cache = True
    ox.settings.log_console = False
    # osmnx's default is 180s (3 minutes) before it gives up on a slow/overloaded
    # Overpass server -- that's a long silent wait for a user staring at a
    # spinner. Cutting it down means we find out something's wrong sooner and
    # can fall back to a different Overpass mirror instead of just waiting.
    ox.settings.requests_timeout = 20
    ox.settings.overpass_url = OVERPASS_MIRRORS[0]
    return True


def fetchFromOverpass(fetchFn, *args, _status=None):
    """Runs an osmnx fetch, and if the default Overpass endpoint fails or times
    out, retries against an independent mirror before giving up. The public
    Overpass instance is a shared, free, frequently-overloaded resource --
    treating one failure as "this campus has no data" (which the code used to
    do, silently) is simply wrong. This surfaces the real failure instead."""
    lastErr = None
    for i, mirror in enumerate(OVERPASS_MIRRORS):
        if _status is not None:
            word = "Trying" if i == 0 else "That was slow -- trying a backup"
            _status.write(f"{word} OpenStreetMap server (up to {ox.settings.requests_timeout}s)...")
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


def throttleNominatim():
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


PHOTON_URL = "https://photon.komoot.io/api/"


def queryPhoton(q, limit=5):
    # Komoot's Photon is a completely separate service/infrastructure built
    # on OSM data -- different servers, different rate-limit bucket from
    # Nominatim. It's not a full replacement (no detailed boundary polygons,
    # just a point + a rough bounding box), but when Nominatim itself is
    # having a bad moment, having ANY independent path to try is the
    # difference between a hard failure and a working, if slightly less
    # precise, result.
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
        extent = props.get("extent")  # Photon: [minLon, maxLat, maxLon, minLat]
        if extent and len(extent) == 4:
            minLon, maxLat, maxLon, minLat = extent
            boundingbox = [str(minLat), str(maxLat), str(minLon), str(maxLon)]

        out.append({
            "class": props.get("osm_key"),
            "type": props.get("osm_value"),
            "display_name": display_name,
            "osm_type": props.get("osm_type"),
            "osm_id": props.get("osm_id"),
            "geojson": None,  # Photon doesn't return full boundary polygons
            "boundingbox": boundingbox,
            "lat": coords[1] if coords else None,
            "lon": coords[0] if coords else None,
        })
    return out


def queryNominatim(q, limit=5, _status=None):
    p = {
        "q": q,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
        "polygon_geojson": 1,
    }

    # NOTE: this used to retry 5 times with backoff (~40s worst case) before
    # giving up. If what we're actually hitting is a STANDING block on this
    # host's shared IP (a real, documented issue -- Nominatim has been known
    # to blanket-throttle Streamlit Community Cloud's egress IPs specifically,
    # because so much hobby traffic hits it from there without following
    # usage policy) then no amount of retrying fixes that, and 40s of
    # retrying before ever trying the fallback is just wasted time. Fail fast
    # instead, and let findCampus() move on to a genuinely different service.
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

        if _status is not None:
            _status.write(f"OpenStreetMap returned HTTP 429 -- retrying in {int(wait)}s...")
        time.sleep(wait)

    # Show the actual evidence instead of a canned guess, so this is
    # verifiable instead of taking my word for it.
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
    # Vectorized version of _friendly(): operates on a whole column at once
    # instead of calling a Python function per row. This -- combined with
    # addLabelAndTrim below -- is the actual fix for the app appearing to
    # freeze on "Fetching buildings and facilities": a dense campus can have
    # many thousands of individually tagged building footprints (far more
    # than road segments), and the old row-by-row .apply() became the real
    # bottleneck at that scale. Non-string values (NaN, None, or an
    # occasional OSM multi-value list) are dropped, matching the original.
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


SIMPLIFY_TOLERANCE = 0.00005  # ~5.5m at typical campus latitudes -- was 4.4m;
# vertex count is the dominant cost in every single render (this map is
# rebuilt from scratch on every sidebar interaction), and sub-2m precision is
# invisible at the zoom levels this app is actually used at


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
    # each layer needs its OWN instance -- reusing one object is a documented
    # folium bug that causes a JS collision and blanks the map
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
    # cheap, purely local computation over data already fetched -- no extra
    # network calls -- so this is essentially free to show whenever a
    # student picks a building
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
    # classic Levenshtein edit distance (single-row DP) -- a freshman on a
    # campus they've never seen doesn't know a building's exact official
    # name or spelling, and typos happen on a phone keyboard constantly.
    # This is what makes "libary" or "student unio" still find the right
    # building instead of returning nothing.
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
        # a direct substring hit is a very strong signal -- rank earlier,
        # tighter matches above later, looser ones
        return 1000 - c.index(q) - abs(len(c) - len(q)) * 0.1
    # typo-tolerant fallback: compare the query against the whole candidate
    # AND against each individual word in it, so "libary" still matches
    # "Central Library (Library)" even though the full strings differ a lot
    words = re.split(r"[\s()]+", c)
    candidates = [w for w in words if w] + [c]
    bestWord, best = min(((w, _editDistance(q, w)) for w in candidates), key=lambda t: t[1])
    # how many edits count as "a plausible typo" scales with word length --
    # 2 edits is reasonable on a 10-letter word, not on a 4-letter one, so a
    # flat cutoff would either miss real typos or return unrelated junk
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
    # walk a GeoJSON geometry's (possibly nested) coordinates to get min/max
    # lon/lat -- works for LineString, MultiLineString, or anything else
    # without caring about the specific nesting depth
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
    # OSM/shapely coordinates default to ~15-17 significant digits when
    # serialized -- way beyond what matters for someone navigating on foot.
    # 5 decimal places is ~1.1m at campus latitudes: comfortably inside a
    # phone's own GPS accuracy, so there's no real-world precision lost, and
    # it shrinks the GeoJSON payload substantially, which directly cuts
    # caching time, network transfer to the browser, and the browser's own
    # JSON-parse + render time.
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
    # Only "Label" is ever read client-side, by the hover tooltip. HasName,
    # Category, and osmid exist purely to help build the search index
    # server-side (see fetchRoadsAndWalkways / namedLocations) and have zero
    # use after that -- shipping them to the browser on every single feature
    # is pure waste, and it adds up fast across thousands of features.
    if not geo:
        return geo
    for feat in geo.get("features", []):
        props = feat.get("properties")
        if props:
            feat["properties"] = {k: v for k, v in props.items() if k in keep}
    return geo


def _segmentEndpoints(geometry):
    # the two (or more, for MultiLineString) endpoint coordinates of a road
    # segment -- used purely to detect "these two OSM ways physically touch",
    # since OSM ways that share a real-world junction share an identical
    # coordinate at that point
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
    """Real campus roads are almost always mapped in OSM as many small,
    disconnected ways -- a driveway stub here, a service-road fork there --
    and typically only ONE of those ways actually carries the `name` tag,
    even though they're all physically the same strip of pavement. Left as
    literal name-tag matches, that means a named road is only ever partly
    searchable and only ever partly highlighted (exactly the two bugs this
    fixes).

    This is a multi-source breadth-first search: start from every segment
    that already has a real name, and spread that name outward through
    directly-touching (shared-endpoint) segments that have NO name of their
    own -- never through a segment that already carries a *different* name,
    so two genuinely different named roads that happen to meet at an
    intersection never bleed into each other. Hop-limited so one named road
    can't accidentally swallow an entire unrelated service-road network many
    junctions away.

    Returns {feature_index: effective_name}."""
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
                    continue  # already named -- its own name, or already claimed
                effectiveName[j] = name
                queue.append((j, hops + 1))

    return effectiveName


def fetchRoadsAndWalkways(polygon_wkt, _status=None):
    # ONE combined Overpass query for both roads and walkways (both are highway=*
    # tags, just different values) instead of two separate round-trips -- OSMnx
    # documents this union behavior explicitly and it's covered by their own test
    # suite, so this is a supported pattern, not a workaround.
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)

    # NOTE: deliberately no blanket try/except swallowing everything here
    # anymore. It used to catch every exception -- including a genuine
    # Overpass timeout on an overloaded public server -- and quietly return
    # empty data, which showed up to the user as the misleading "OSM has no
    # data for this campus" even though the real problem was the network
    # request failing. Real failures now propagate with their real message.
    gdf = fetchFromOverpass(ox.features_from_polygon, poly, {"highway": WALKWAY_VALUES + ROAD_VALUES}, _status=_status)
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

    # named-road index built directly from the SAME GeoJSON that gets
    # rendered/tooltipped on the map (not a separate pre-trim pass over the
    # raw dataframe) -- that guarantees every road the map draws is
    # guaranteed to be consistent with what's searchable. Names are
    # propagated across touching unnamed segments first (see
    # _propagateRoadNames) so a road split into many partially-tagged OSM
    # ways is treated -- and highlighted -- as the single connected road
    # it actually is, not just whichever fragment happened to carry the tag.
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

    # done building the search index off HasName/Label -- now drop every
    # property except Label from what actually gets shipped to render
    roadGeo = _stripUnusedProps(roadGeo)
    walkGeo = _stripUnusedProps(walkGeo)

    return roadGeo, walkGeo, namedRoads, namedRoadGeo
fetchRoadsAndWalkways = st.cache_data(show_spinner=False, ttl="24h")(fetchRoadsAndWalkways)


def fetchBuildingsAndFacilities(polygon_wkt, _status=None):
    # ONE combined query for buildings + facilities instead of two round-trips.
    # a feature CAN legitimately have both building=yes and amenity=library --
    # that's fine, it lands in both splits below, and the existing dedup step
    # (stripDuplicateBuildings) already handles exactly that overlap correctly.
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)

    gdf = fetchFromOverpass(ox.features_from_polygon, poly, {
        "building": True,
        "amenity": FACILITY_AMENITY_VALUES,
        "leisure": FACILITY_LEISURE_VALUES,
    }, _status=_status)
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
    # common naming confusion: "University of X" vs the institution's real name
    # "X University" (and same for College) -- try the swap as a genuine second
    # attempt rather than assuming either order is always correct
    m = re.match(r'^(university|college)\s+of\s+(.+)$', name.strip(), re.IGNORECASE)
    if not m:
        return None
    kind, rest = m.groups()
    return f"{rest.strip()} {kind.title()}"


def findCampus(name, _status=None):
    try:
        results = queryNominatim(name, _status=_status)
    except ValueError as nomErr:
        # Nominatim's search step itself is failing (e.g. rate-limited) even
        # after real retries -- try a fully independent service before
        # giving up entirely, instead of failing outright on one provider
        # having a bad moment
        if _status is not None:
            _status.write("OpenStreetMap search is unavailable -- trying a backup search service...")
        try:
            results = queryPhoton(name)
        except Exception:
            raise nomErr  # the original Nominatim error is more specific -- surface that, not the backup's

    if not results:
        raise ValueError(f'No results found for **"{name}"** on OpenStreetMap.\n\nTry a more specific name, e.g. `"{name}, City, Country"`')

    results.sort(key=lambda r: not looksLikeCampus(r))
    edu_hits = [r for r in results if looksLikeCampus(r)]

    if not edu_hits:
        alt = _alternateNameGuess(name)
        if alt:
            try:
                altResults = queryNominatim(alt, _status=_status)
                altResults.sort(key=lambda r: not looksLikeCampus(r))
                altEduHits = [r for r in altResults if looksLikeCampus(r)]
                if altEduHits:
                    edu_hits = altEduHits
                    results = altResults
            except Exception:
                pass  # if the alternate attempt fails for any reason, fall through to the honest error below

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

    top_hit = edu_hits[0]
    hitName = top_hit.get("display_name", name)

    # Prefer a direct OSM-ID lookup over re-searching by name: it's the exact
    # same place we already matched, hits Nominatim's lighter "lookup"
    # endpoint instead of a fresh fuzzy "search", and avoids burning a second
    # full search request against an already-strained rate limit.
    osmType = top_hit.get("osm_type")
    osmId = top_hit.get("osm_id")
    typePrefix = {"node": "N", "way": "W", "relation": "R"}.get(osmType)

    gdf = None
    boundaryErr = None
    try:
        throttleNominatim()
        if typePrefix and osmId:
            gdf = ox.geocode_to_gdf(f"{typePrefix}{osmId}", by_osmid=True)
        else:
            gdf = ox.geocode_to_gdf(hitName)
    except Exception as e:
        boundaryErr = e

    if boundaryErr is not None or gdf is None or gdf.empty or gdf.iloc[0].geometry.geom_type not in ("Polygon", "MultiPolygon"):
        # the precise boundary lookup failed or came back unusable -- rather
        # than a hard failure, fall back to the bounding box the search step
        # already gave us for free (Nominatim includes one on every result,
        # no extra request needed). It's a rectangle, not the campus's real
        # outline, so it's a genuine downgrade -- but a slightly-imprecise
        # working map beats no map, especially when the reason we're here is
        # that a second network request just got rate-limited.
        bbox = top_hit.get("boundingbox")
        if bbox and len(bbox) == 4:
            try:
                south, north, west, east = (float(v) for v in bbox)
                g = shapelyBox(west, south, east, north)
                if _status is not None:
                    _status.write(f"Using an approximate boundary for {hitName} -- the precise outline "
                                   f"was temporarily unavailable.")
                return hitName, g.wkt
            except Exception:
                pass

        if boundaryErr is not None:
            raise ValueError(
                f'Found **"{hitName}"** but could not get its precise boundary, and no '
                f'fallback bounding box was available either.\n\n'
                f'Raw error: `{boundaryErr}`'
            )
        raise ValueError(
            f'**"{hitName}"** is in OSM but only as a point, not a boundary polygon.\n\n'
            f'Try a more specific search or check [openstreetmap.org](https://www.openstreetmap.org).'
        )

    g = gdf.iloc[0].geometry
    if not g.is_valid:
        g = g.buffer(0)
    return hitName, g.wkt
findCampus = st.cache_data(show_spinner=False, ttl="24h")(findCampus)


def prepareCampusData(polygon_wkt, active_layers, status):
    """Fetch + shape everything the map needs. Pure data, no folium objects --
    this is the only part that's slow (network calls), and it's cached by
    fetchRoadsAndWalkways / fetchBuildingsAndFacilities so it only runs once
    per campus, not on every sidebar interaction."""
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)
    minx, miny, maxx, maxy = poly.bounds

    layerData = {}

    # NOTE: always fetch + store BOTH members of each pair, regardless of which
    # boxes are checked right now. The Overpass query already pulls both
    # members together (it's one combined query), so this costs nothing extra
    # over the network -- and it's the fix for sidebar toggles doing nothing:
    # previously the untouched half of a pair was fetched and immediately
    # thrown away, so checking it later had no data to show without a full
    # re-fetch. Now everything is kept, and which layers are actually drawn is
    # decided fresh at render time from the live checkbox state.
    #
    # NOTE: these run sequentially, not concurrently. A prior version ran them
    # in a ThreadPoolExecutor to cut latency, but Streamlit's own caching and
    # session-context machinery isn't reliably safe to call from a background
    # thread it doesn't know about -- that's exactly what caused the app to
    # hang indefinitely instead of ever finishing or timing out. A predictable
    # ~30-60s beats an unpredictable infinite hang.
    status.update(label="Fetching roads and pedestrian paths...")
    try:
        roadGeo, walkGeo, namedRoads, namedRoadGeo = fetchRoadsAndWalkways(polygon_wkt, _status=status)
    except Exception as e:
        raise ValueError(
            f"Couldn't fetch road/path data from OpenStreetMap's Overpass service.\n\n{e}"
        )
    layerData["roads"] = roadGeo
    layerData["walkways"] = walkGeo
    status.write(f"Roads: {len(roadGeo['features']) if roadGeo else 0} segments, "
                 f"Paths: {len(walkGeo['features']) if walkGeo else 0} segments")

    status.update(label="Fetching buildings and facilities...")
    try:
        bldGdf, facGdf = fetchBuildingsAndFacilities(polygon_wkt, _status=status)
    except Exception as e:
        raise ValueError(
            f"Couldn't fetch building data from OpenStreetMap's Overpass service.\n\n{e}"
        )

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
        raise ValueError("OSM has no tagged data for this campus. Try a different campus or check openstreetmap.org.")

    status.write("Indexing named buildings and roads for search...")

    # named-building index for the "Find a building" search -- real OSM names only,
    # decorated with category so searching "library" or "cafe" surfaces matches
    # even when the freshman doesn't know the specific building's name
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
    # bottom-right: clear of the zoom control (top-left), Fullscreen button
    # (top-right), and our own legend (bottom-left)
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
    """Builds a brand new folium map every call. Cheap: no network calls, just
    re-attaches already-fetched GeoJSON. Building fresh (instead of mutating
    a single map object stored across reruns) is what keeps this fast --
    a reused, endlessly-mutated map object accumulates every fit_bounds/marker/
    highlight ever added across the whole session and grinds the browser
    to a halt after a few clicks.

    visibleLayers is read fresh from the sidebar checkboxes on every single
    call, so which layers actually get drawn always matches their current
    state -- this is what makes the sidebar toggles work, since layerData
    itself always contains everything that was ever fetched."""
    miny, minx, maxy, maxx = bounds
    cLat = (miny + maxy) / 2
    cLon = (minx + maxx) / 2

    m = leafmap.Map(
        center=[cLat, cLon],
        zoom=15,
        tiles="CartoDB.Positron",
        prefer_canvas=True,   # canvas renderer instead of SVG -- much faster
                               # with hundreds/thousands of building & road
                               # shapes, which is exactly this app's workload
        control_scale=True,   # small distance scale bar -- helps students
                               # gauge how far a walk actually is
    )

    folium_plugins.Fullscreen(
        position="topright",
        title="Expand map",
        title_cancel="Exit fullscreen",
        force_separate_button=True
    ).add_to(m)

    # "show my location" -- uses the browser's own GPS/Wi-Fi location, no
    # server round-trip, so it's essentially free and is the single most
    # useful feature for a freshman who is actually lost on campus right now
    folium_plugins.LocateControl(
        position="topright",
        strings={"title": "Show my location"},
        flyTo=True,
    ).add_to(m)

    # lets a student drag out a line/area on the map to see the real
    # walking distance, e.g. "is it faster to cut through the quad?"
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
            smooth_factor=1.5,  # a bit more render-time line simplification in
                                 # Leaflet itself -- imperceptible at the zoom
                                 # levels this app is used at, cheaper to draw
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
# Dialed WAY back after repeatedly breaking Streamlit's own internals (icon
# rendering, label contrast) by overriding its default component styling.
# Streamlit's default light theme is already clean, accessible, and battle
# tested -- this only touches things that are genuinely isolated and can't
# fight with Streamlit's own CSS: a font import, a header banner, and the
# primary button's color. Nothing here touches the sidebar background,
# generic text/span/div elements, or any Streamlit internals.
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
            campusName, campusPoly = findCampus(searchTerm, _status=status)
            status.update(label=f"Found: {campusName}", state="running")
            status.write(f"Matched: {campusName}")
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

    # force an immediate follow-up rerun so the sidebar picks up the freshly
    # populated namedLocations/namedRoads on its next execution -- otherwise the
    # "Find a building"/"Find a road" dropdowns wouldn't appear until some
    # unrelated later interaction happened to trigger another rerun
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

# a brand new map is built on every run instead of reusing/mutating one
# stored object -- that's what keeps this fast and keeps focus/highlight
# state correct no matter which building or road was picked last
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
