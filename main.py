import os
import json
import tempfile
import time
import ee
from fastapi import FastAPI

app = FastAPI()

CACHE = {
    "rows": None,
    "updated": 0
}

CACHE_SECONDS = 6 * 60 * 60  # 6 hours


def init_gee():
    service_account = os.environ["GEE_SERVICE_ACCOUNT"]
    key_json = os.environ["GEE_PRIVATE_KEY_JSON"]
    project_id = os.environ["GEE_PROJECT_ID"]

    key_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    key_file.write(key_json.encode("utf-8"))
    key_file.close()

    credentials = ee.ServiceAccountCredentials(service_account, key_file.name)
    ee.Initialize(credentials, project=project_id)


@app.on_event("startup")
def startup():
    init_gee()


@app.get("/")
def home():
    return {
        "status": "Render + Earth Engine running",
        "endpoints": ["/stress", "/stress-compact", "/refresh"]
    }


def compute_stress():
    states = ee.FeatureCollection("FAO/GAUL/2015/level1")
    districtsAll = ee.FeatureCollection("FAO/GAUL/2015/level2")

    tamilNadu = states \
        .filter(ee.Filter.eq("ADM0_NAME", "India")) \
        .filter(ee.Filter.eq("ADM1_NAME", "Tamil Nadu"))

    districts = districtsAll \
        .filter(ee.Filter.eq("ADM0_NAME", "India")) \
        .filter(ee.Filter.eq("ADM1_NAME", "Tamil Nadu"))

    currentStart = "2026-01-01"
    currentEnd = "2026-03-31"
    baselineYears = [2021, 2022, 2023, 2024, 2025]

    def getS1(startDate, endDate):
        return ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterBounds(tamilNadu) \
            .filterDate(startDate, endDate) \
            .filter(ee.Filter.eq("instrumentMode", "IW")) \
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
            .select(["VV", "VH"])

    def despeckle(img):
        vv = img.select("VV").focal_median(radius=30, units="meters")
        vh = img.select("VH").focal_median(radius=30, units="meters")
        return ee.Image.cat([vv, vh]).copyProperties(img, img.propertyNames())

    def addMoistureProxy(img):
        vv_db = img.select("VV")
        vh_db = img.select("VH")

        vv_lin = ee.Image(10).pow(vv_db.divide(10))
        vh_lin = ee.Image(10).pow(vh_db.divide(10))

        ratio = vh_lin.divide(vv_lin).rename("vh_vv_ratio")
        return img.addBands(ratio)

    currentCollection = getS1(currentStart, currentEnd) \
        .map(despeckle) \
        .map(addMoistureProxy)

    currentRatio = currentCollection.median() \
        .clip(tamilNadu) \
        .select("vh_vv_ratio")

    def seasonalComposite(year):
        year = ee.Number(year)
        start = ee.Date.fromYMD(year, 1, 1)
        end = ee.Date.fromYMD(year, 3, 31)

        return getS1(start, end) \
            .map(despeckle) \
            .map(addMoistureProxy) \
            .median() \
            .select("vh_vv_ratio") \
            .clip(tamilNadu) \
            .set("year", year)

    baselineCollection = ee.ImageCollection.fromImages(
        [seasonalComposite(y) for y in baselineYears]
    )

    baselineMean = baselineCollection.mean()
    baselineStd = baselineCollection.reduce(ee.Reducer.stdDev())

    rawStress = baselineMean.subtract(currentRatio)
    safeStd = baselineStd.max(0.001)

    stress = rawStress.divide(safeStd) \
        .focal_mean(radius=1000, units="meters") \
        .clamp(-3, 3) \
        .clip(tamilNadu) \
        .rename("stress")

    highStress = stress.gt(1).rename("high_stress")

    districtStats = stress.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.max(),
            sharedInputs=True
        ),
        scale=3000,
        tileScale=4
    )

    highStats = highStress.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=3000,
        tileScale=4
    )

    def join_row(f):
        code = f.get("ADM2_CODE")
        match = highStats.filter(ee.Filter.eq("ADM2_CODE", code)).first()
        frac = ee.Algorithms.If(match, match.get("mean"), None)

        return ee.Feature(None, {
            "district": f.get("ADM2_NAME"),
            "mean_stress": f.get("mean"),
            "max_stress": f.get("max"),
            "high_pct": ee.Algorithms.If(
                frac,
                ee.Number(frac).multiply(100),
                None
            )
        })

    table = districtStats.map(join_row) \
        .filter(ee.Filter.notNull(["mean_stress", "max_stress", "high_pct"])) \
        .sort("high_pct", False)

    data = table.limit(20).getInfo()

    rows = []

    for f in data["features"]:
        p = f["properties"]

        rows.append({
            "district": p["district"],
            "high_pct": round(p["high_pct"], 1),
            "mean_stress": round(p["mean_stress"], 2),
            "max_stress": round(p["max_stress"], 2)
        })

    return rows


def get_cached_rows(force=False):
    now = time.time()

    if force or CACHE["rows"] is None or now - CACHE["updated"] > CACHE_SECONDS:
        CACHE["rows"] = compute_stress()
        CACHE["updated"] = now

    return CACHE["rows"]


@app.get("/stress")
def stress():
    rows = get_cached_rows()
    return {
        "updated": CACHE["updated"],
        "count": len(rows),
        "rows": rows
    }


@app.get("/stress-compact")
def stress_compact():
    rows = get_cached_rows()

    compact = []
    for r in rows[:8]:
        compact.append({
            "d": r["district"],
            "h": r["high_pct"]
        })

    return compact


@app.get("/refresh")
def refresh():
    rows = get_cached_rows(force=True)
    return {
        "status": "refreshed",
        "count": len(rows),
        "rows": rows
    }