"""
CampusWay — Nominatim debug tool
Deploy this as a SEPARATE Streamlit app temporarily (e.g. debug_search.py).
It makes the exact same Nominatim call the main app does, with zero caching,
and shows every field of every result so we can see why looksLikeCampus fails.
"""
import streamlit as st
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
req_headers = {"User-Agent": "global-campus-navigator/1.0 (streamlit-app)"}

NOT_CAMPUS = {"leisure", "shop", "tourism", "highway", "natural"}
EDU_TAG_PAIRS = {
    ("amenity", "university"),
    ("amenity", "college"),
    ("amenity", "school"),
    ("amenity", "research_institute"),
    ("landuse", "education"),
    ("boundary", "educational"),
    ("place", "campus"),
}
campusNameHints = (
    "university", "college", "institute of technology", "polytechnic",
    "academy", "ecole", "universidad", "universita", "mit", "caltech",
)

def looksLikeCampus(r):
    pair = (r.get("class"), r.get("type"))
    if pair in EDU_TAG_PAIRS:
        return True, f"EDU_TAG_PAIRS hit: {pair}"
    if r.get("class") in NOT_CAMPUS:
        return False, f"NOT_CAMPUS block: class={r.get('class')!r}"
    dn = (r.get("display_name") or "").lower()
    matched = [h for h in campusNameHints if h in dn]
    if matched:
        return True, f"display_name hint: {matched}"
    return False, f"no match (class={r.get('class')!r}, type={r.get('type')!r}, no hint in display_name)"

st.title("CampusWay — Nominatim Debug")
query = st.text_input("Search term", value="MIT")

if st.button("Run") and query.strip():
    params = {
        "q": query.strip(),
        "format": "jsonv2",
        "limit": 5,
        "addressdetails": 1,
        "polygon_geojson": 0,   # skip geometry to keep output readable
    }
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=req_headers, timeout=10)
        st.write(f"**HTTP status:** {r.status_code}")
        results = r.json()
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    if not results:
        st.warning("Nominatim returned an **empty list** — no results at all.")
        st.stop()

    st.write(f"**{len(results)} result(s) returned:**")
    for i, res in enumerate(results):
        passes, reason = looksLikeCampus(res)
        color = "🟢" if passes else "🔴"
        with st.expander(f"{color} [{i}] {res.get('display_name', '')[:80]}"):
            st.write({
                "class (json)":    res.get("class"),
                "category (jsonv2)": res.get("category"),
                "type":            res.get("type"),
                "osm_type":        res.get("osm_type"),
                "osm_id":          res.get("osm_id"),
                "geojson type":    (res.get("geojson") or {}).get("type"),
                "looksLikeCampus": passes,
                "reason":          reason,
                "display_name":    res.get("display_name"),
            })
