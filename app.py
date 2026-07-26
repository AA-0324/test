import streamlit as st

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

import time
import html
import re
import requests
import pandas as pd
import geopandas as gpd
import folium
from branca.element import MacroElement, Template
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
    "buildings":  styleFor("#1f77b4", "#1f77b4", 0.45, 1.0),
    "walkways":   styleFor("#2ca02c", "#2ca02c", 0.0,  2.5, dashed=True),
    "roads":      styleFor("#7f7f7f", "#7f7f7f", 0.0,  1.5),
    "facilities": styleFor("#d62728", "#d62728", 0.7,  1.5),
}

LAYER_COLORS = {
    "buildings":  "#1f77b4",
    "walkways":   "#2ca02c",
    "roads":      "#7f7f7f",
    "facilities": "#d62728",
}

req_headers = {"User-Agent": "global-campus-navigator/1.0 (streamlit-app)"}
RATE_LIMIT_GAP = 1.5


@st.cache_resource
def initOsmnx():
    ox.settings.use_cache = True
    ox.settings.log_console = False
    return True


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


def queryNominatim(q, limit=5):
    p = {
        "q": q,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
        "polygon_geojson": 1,
    }
    throttleNominatim()
    r = requests.get(NOMINATIM_URL, params=p, headers=req_headers, timeout=10)

    if r.status_code == 429:
        time.sleep(3)
        r = requests.get(NOMINATIM_URL, params=p, headers=req_headers, timeout=10)

    if r.status_code == 429:
        raise ValueError(
            "OpenStreetMap's free search service is rate-limiting requests right now "
            "(HTTP 429). This is common on shared cloud hosting and isn't caused by "
            "this app directly. It usually clears on its own -- try again shortly."
        )

    r.raise_for_status()
    return r.json()


FALLBACK_TAG = {
    "buildings":  ("building", "Building"),
    "facilities": ("amenity", "Facility"),
    "walkways":   ("highway", "Path"),
    "roads":      ("highway", "Road"),
}


USELESS_CATEGORY_VALUES = {"yes", "university", "college"}


def _friendly(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.lower() in USELESS_CATEGORY_VALUES:
        return None
    return value.replace("_", " ").replace("-", " ").title()


def _labelFor(row, layer_key):
    name = row.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip(), True
    tag_col, fallback_word = FALLBACK_TAG[layer_key]
    val = row.get(tag_col)
    if layer_key == "facilities" and not _friendly(val):
        val = row.get("leisure")
    friendly = _friendly(val)
    return (friendly if friendly else fallback_word), False


def _categoryFor(row, layer_key):
    tag_col, _ = FALLBACK_TAG[layer_key]
    val = row.get(tag_col)
    if layer_key == "facilities" and not _friendly(val):
        val = row.get("leisure")
    return _friendly(val)


def addLabelAndTrim(gdf, layer_key):
    if gdf is None or gdf.empty:
        return gdf
    geom_col = gdf.geometry.name
    keep_cols = [geom_col]
    if "osmid" in gdf.columns:
        keep_cols.append("osmid")
    try:
        gdf = gdf.copy()
        labels = gdf.apply(lambda row: _labelFor(row, layer_key), axis=1)
        gdf["Label"]    = labels.apply(lambda t: t[0])
        gdf["HasName"]  = labels.apply(lambda t: t[1])
        gdf["Category"] = gdf.apply(lambda row: _categoryFor(row, layer_key), axis=1)
        keep_cols += ["Label", "HasName", "Category"]
        return gdf[keep_cols]
    except Exception:
        return gdf


SIMPLIFY_TOLERANCE = 0.000015  # ~1.5m at typical campus latitudes -- conservative enough to not visibly distort shapes


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


def fetchRoadsAndWalkways(polygon_wkt):
    # ONE combined Overpass query for both roads and walkways (both are highway=*
    # tags, just different values) instead of two separate round-trips -- OSMnx
    # documents this union behavior explicitly and it's covered by their own test
    # suite, so this is a supported pattern, not a workaround.
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)
    try:
        gdf = ox.features_from_polygon(poly, tags={"highway": WALKWAY_VALUES + ROAD_VALUES})
        if gdf is None or gdf.empty or "highway" not in gdf.columns:
            return None, None, {}
        gdf = gdf.to_crs("EPSG:4326") if gdf.crs else gdf
        gdf = _simplify(gdf)

        walkways = gdf[gdf["highway"].isin(WALKWAY_VALUES)].copy()
        roads = gdf[gdf["highway"].isin(ROAD_VALUES)].copy()

        # named-road index: group by name and combine bounds across every segment
        # sharing that name -- a single named road is often split into many
        # disconnected OSM ways, so this must be a union of all of them, not just
        # the first one found (unlike buildings, which are single polygons)
        namedRoads = {}
        namedRoadGeo = {}  # GeoJSON segments for highlighted rendering
        for segment_gdf in (roads, walkways):
            if segment_gdf.empty or "name" not in segment_gdf.columns:
                continue
            named_segments = segment_gdf[segment_gdf["name"].notna()]
            for name, group in named_segments.groupby("name"):
                name = str(name).strip()
                if not name:
                    continue
                minx, miny, maxx, maxy = group.total_bounds
                namedRoads[name] = _mergeBounds(namedRoads.get(name), miny, minx, maxy, maxx)
                # accumulate GeoJSON for highlight layer
                geo_chunk = group.__geo_interface__
                if name not in namedRoadGeo:
                    namedRoadGeo[name] = geo_chunk
                else:
                    # merge feature lists
                    existing_feats = namedRoadGeo[name].get("features", [])
                    new_feats = geo_chunk.get("features", [])
                    namedRoadGeo[name] = {"type": "FeatureCollection", "features": existing_feats + new_feats}

        walkways = addLabelAndTrim(walkways, "walkways")
        roads = addLabelAndTrim(roads, "roads")
        walkGeo = walkways.__geo_interface__ if walkways is not None and not walkways.empty else None
        roadGeo = roads.__geo_interface__ if roads is not None and not roads.empty else None
        return roadGeo, walkGeo, namedRoads, namedRoadGeo
    except Exception:
        return None, None, {}, {}
fetchRoadsAndWalkways = st.cache_data(show_spinner=False, ttl="6h")(fetchRoadsAndWalkways)


def fetchBuildingsAndFacilities(polygon_wkt):
    # ONE combined query for buildings + facilities instead of two round-trips.
    # a feature CAN legitimately have both building=yes and amenity=library --
    # that's fine, it lands in both splits below, and the existing dedup step
    # (stripDuplicateBuildings) already handles exactly that overlap correctly.
    from shapely import wkt as swkt
    poly = swkt.loads(polygon_wkt)
    try:
        gdf = ox.features_from_polygon(poly, tags={
            "building": True,
            "amenity": FACILITY_AMENITY_VALUES,
            "leisure": FACILITY_LEISURE_VALUES,
        })
        if gdf is None or gdf.empty:
            return None, None
        gdf = gdf.to_crs("EPSG:4326") if gdf.crs else gdf
        gdf = _simplify(gdf)

        has_building = gdf["building"].notna() if "building" in gdf.columns else pd.Series(False, index=gdf.index)
        has_amenity = gdf["amenity"].isin(FACILITY_AMENITY_VALUES) if "amenity" in gdf.columns else pd.Series(False, index=gdf.index)
        has_leisure = gdf["leisure"].isin(FACILITY_LEISURE_VALUES) if "leisure" in gdf.columns else pd.Series(False, index=gdf.index)

        buildings = gdf[has_building].copy()
        facilities = gdf[has_amenity | has_leisure].copy()

        buildings = addLabelAndTrim(buildings, "buildings")
        facilities = addLabelAndTrim(facilities, "facilities")
        return buildings, facilities
    except Exception:
        return None, None
fetchBuildingsAndFacilities = st.cache_data(show_spinner=False, ttl="6h")(fetchBuildingsAndFacilities)


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


def findCampus(name):
    results = queryNominatim(name)
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
    namedRoads = {}
    namedRoadGeo = {}

    if "roads" in active_layers or "walkways" in active_layers:
        status.update(label="Fetching roads and pedestrian paths...")
        roadGeo, walkGeo, namedRoads, namedRoadGeo = fetchRoadsAndWalkways(polygon_wkt)
        if "roads" in active_layers:
            layerData["roads"] = roadGeo
        if "walkways" in active_layers:
            layerData["walkways"] = walkGeo
        # namedRoadGeo is populated inside fetchRoadsAndWalkways above

    bldGdf = facGdf = None
    if "buildings" in active_layers or "facilities" in active_layers:
        status.update(label="Fetching buildings and facilities...")
        bldGdf, facGdf = fetchBuildingsAndFacilities(polygon_wkt)
        if "buildings" in active_layers:
            bldGdf = stripDuplicateBuildings(bldGdf, facGdf)
            layerData["buildings"] = bldGdf.__geo_interface__ if bldGdf is not None and not bldGdf.empty else None
        if "facilities" in active_layers:
            layerData["facilities"] = facGdf.__geo_interface__ if facGdf is not None and not facGdf.empty else None

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
        folium.GeoJson(
            data=geo,
            name=layer_labels[k],
            style_function=campusStyles[k],
            tooltip=makeTooltip(),
        ).add_to(m)
        foundAnything = True

    if not foundAnything:
        raise ValueError("OSM has no tagged data for this campus. Try a different campus or check openstreetmap.org.")

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

    m.fit_bounds([[miny, minx], [maxy, maxx]])

    rendered = [k for k in drawOrder if counts.get(k, 0) > 0]
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
                    background: white; padding: 8px 12px; border-radius: 4px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 13px;
                    font-family: sans-serif; line-height: 1.4;">
            {rows}
        </div>
        {{% endmacro %}}
        """
        legend = MacroElement()
        legend._template = Template(legend_html)
        m.get_root().add_child(legend)

    return m, counts, namedLocations, namedRoads, namedRoadGeo


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

    if namedLocs or namedRds:
        st.divider()
        st.subheader("Navigate")
        st.caption("Type a building or road name to jump straight to it.")

        if namedLocs:
            selected = st.selectbox(
                "Find a building",
                sorted(namedLocs.keys()),
                index=None,
                placeholder="Type to search buildings...",
            )
            if selected:
                focusName = selected

        if namedRds:
            selectedRoad = st.selectbox(
                "Find a road",
                sorted(namedRds.keys()),
                index=None,
                placeholder="Type to search roads...",
            )
            if selectedRoad:
                focusRoad = selectedRoad


# ── main area ─────────────────────────────────────────────────────────────────

if searchBtn and campusInput.strip():
    newSearch = campusInput.strip()
    if newSearch != st.session_state.get("lastSearch", ""):
        st.session_state["lastSearch"] = newSearch
        st.session_state.pop("campusMap", None)
        st.session_state.pop("namedLocations", None)
        st.session_state.pop("namedRoads", None)
        st.session_state.pop("namedRoadGeo", None)

searchTerm = st.session_state.get("lastSearch", "")

if not searchTerm:
    st.info("Enter a university or college name in the sidebar and click **Generate Map** to get started.")
    st.stop()

if "campusMap" not in st.session_state:
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
                campusMap, layerCounts, namedLocations, namedRoads, namedRoadGeo = buildCampusMap(campusPoly, active_layers, status)
                st.session_state["campusMap"] = campusMap
                st.session_state["namedLocations"] = namedLocations
                st.session_state["namedRoads"] = namedRoads
                st.session_state["namedRoadGeo"] = namedRoadGeo
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

    # force an immediate follow-up rerun so the sidebar picks up the freshly
    # populated namedLocations/namedRoads on its next execution -- otherwise the
    # "Find a building"/"Find a road" dropdowns wouldn't appear until some
    # unrelated later interaction happened to trigger another rerun
    st.rerun()

campusMap = st.session_state["campusMap"]

if focusName and focusName in st.session_state.get("namedLocations", {}):
    flat, flon = st.session_state["namedLocations"][focusName]
    pad = 0.0012
    campusMap.fit_bounds([[flat - pad, flon - pad], [flat + pad, flon + pad]])
    folium.CircleMarker(
        location=[flat, flon],
        radius=16,
        color="#ff6600",
        weight=3,
        fill=True,
        fill_color="#ff6600",
        fill_opacity=0.15,
        tooltip=folium.Tooltip(focusName),
    ).add_to(campusMap)
elif focusRoad and focusRoad in st.session_state.get("namedRoads", {}):
    s, w, n, e = st.session_state["namedRoads"][focusRoad]
    padLat = max((n - s) * 0.15, 0.0005)
    padLon = max((e - w) * 0.15, 0.0005)
    campusMap.fit_bounds([[s - padLat, w - padLon], [n + padLat, e + padLon]])
    # highlight the matched road segments with a vivid overlay
    roadGeoData = st.session_state.get("namedRoadGeo", {}).get(focusRoad)
    if roadGeoData:
        folium.GeoJson(
            data=roadGeoData,
            name="__highlight__",
            style_function=lambda feat: {
                "color": "#ff6600",
                "weight": 6,
                "opacity": 0.85,
            },
            tooltip=folium.Tooltip(focusRoad),
        ).add_to(campusMap)

# ── CampusWay branding (map overlay) ─────────────────────────────────────────
branding_html = """
{% macro html(this, kwargs) %}
<div style="
    position: fixed;
    top: 12px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 9999;
    background: rgba(255,255,255,0.92);
    backdrop-filter: blur(4px);
    padding: 5px 14px;
    border-radius: 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.18);
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #1a1a2e;
    user-select: none;
    pointer-events: none;
">
    <span style="color:#1f77b4;">Campus</span><span style="color:#ff6600;">Way</span>
</div>
{% endmacro %}
"""
branding = MacroElement()
branding._template = Template(branding_html)
campusMap.get_root().add_child(branding)

campusMap.to_streamlit(height=620)
