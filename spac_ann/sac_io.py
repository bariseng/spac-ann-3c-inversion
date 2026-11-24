"""Zero-dependency binary SAC header/trace reader and WGS84 geodesic utilities."""
from __future__ import annotations

import datetime as dt
import os
import struct
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from pyproj import Geod, Transformer

GEOD = Geod(ellps="WGS84")


@dataclass
class SacHeader:
    path: str
    endian: str
    delta: float
    b: float
    e: float
    stla: float
    stlo: float
    stel: float
    npts: int
    nzyear: int
    nzjday: int
    nzhour: int
    nzmin: int
    nzsec: int
    nzmsec: int
    kstnm: str
    kcmpnm: str
    knetwk: str
    start_utc: dt.datetime
    end_utc: dt.datetime
    duration_s: float
    fs: float
    device_id: str
    component: str


def _clean_sac_str(raw: bytes) -> str:
    s = raw.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
    return "" if s == "-12345" else s


def read_sac_header(path: str) -> SacHeader:
    """Read a binary SAC header (auto-detecting little/big endian)."""
    with open(path, "rb") as f:
        hdr = f.read(632)
    if len(hdr) < 632:
        raise ValueError(f"Truncated SAC file: {path}")

    i_le = struct.unpack("<40i", hdr[280:440])
    i_be = struct.unpack(">40i", hdr[280:440])
    if i_le[6] == 6:
        endian = "<"
        il = i_le
        fl = struct.unpack("<70f", hdr[:280])
    elif i_be[6] == 6:
        endian = ">"
        il = i_be
        fl = struct.unpack(">70f", hdr[:280])
    else:
        raise ValueError(f"Invalid SAC NVHDR in {path}: le={i_le[6]}, be={i_be[6]}")

    delta = float(fl[0])
    fs = round(1.0 / delta, 4) if delta > 0 else 0.0
    if abs(fs - round(fs)) < 1e-3 and fs > 0:
        fs = float(round(fs))
        delta = 1.0 / fs

    b = float(fl[5])
    e = float(fl[6])
    stla = float(fl[31])
    stlo = float(fl[32])
    stel = float(fl[33])
    nzyear, nzjday, nzhour, nzmin, nzsec, nzmsec = il[0], il[1], il[2], il[3], il[4], il[5]
    npts = int(il[9])

    kstnm = _clean_sac_str(hdr[440:448])
    kcmpnm = _clean_sac_str(hdr[600:608])
    knetwk = _clean_sac_str(hdr[608:616])

    ref_utc = dt.datetime(nzyear, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        days=nzjday - 1,
        hours=nzhour,
        minutes=nzmin,
        seconds=nzsec,
        milliseconds=nzmsec,
    )
    start_utc = ref_utc + dt.timedelta(seconds=b)
    duration_s = (npts - 1) * delta if npts > 1 else 0.0
    end_utc = start_utc + dt.timedelta(seconds=duration_s)

    fname = os.path.basename(path)
    device_id = fname.split(".")[0] if "." in fname else kstnm
    comp = ""
    if kcmpnm and kcmpnm[-1] in ("E", "N", "Z"):
        comp = kcmpnm[-1]
    else:
        parts = fname.split(".")
        if len(parts) >= 2 and parts[-2] in ("E", "N", "Z"):
            comp = parts[-2]

    return SacHeader(
        path=path,
        endian=endian,
        delta=delta,
        b=b,
        e=e,
        stla=stla,
        stlo=stlo,
        stel=stel,
        npts=npts,
        nzyear=nzyear,
        nzjday=nzjday,
        nzhour=nzhour,
        nzmin=nzmin,
        nzsec=nzsec,
        nzmsec=nzmsec,
        kstnm=kstnm,
        kcmpnm=kcmpnm,
        knetwk=knetwk,
        start_utc=start_utc,
        end_utc=end_utc,
        duration_s=duration_s,
        fs=fs,
        device_id=device_id,
        component=comp,
    )


def read_sac(path: str) -> Tuple[SacHeader, np.ndarray]:
    """Read SAC header and float64 trace data."""
    hdr = read_sac_header(path)
    dtype = np.dtype(hdr.endian + "f4")
    with open(path, "rb") as f:
        f.seek(632)
        data = np.fromfile(f, dtype=dtype, count=hdr.npts).astype(np.float64)
    return hdr, data


def slice_sac_window(
    hdr: SacHeader,
    data: np.ndarray,
    t_start: dt.datetime,
    t_end: dt.datetime,
) -> np.ndarray:
    """Extract exact time window [t_start, t_end) from trace."""
    i0 = int(round((t_start - hdr.start_utc).total_seconds() * hdr.fs))
    n_req = int(round((t_end - t_start).total_seconds() * hdr.fs))
    if i0 < 0 or i0 + n_req > len(data):
        raise ValueError(
            f"Requested window [{t_start}, {t_end}] out of bounds for {hdr.path} "
            f"[{hdr.start_utc}, {hdr.end_utc}] (i0={i0}, n_req={n_req}, len={len(data)})"
        )
    return data[i0 : i0 + n_req].copy()


def geodesic_dist_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return WGS84 geodesic distance in meters between two geographic points."""
    _, _, d = GEOD.inv(lon1, lat1, lon2, lat2)
    return float(d)


def to_local_xy(
    lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert WGS84 lat/lon to local ENU meters in auto-detected UTM relative to (lat0, lon0)."""
    utm_zone = int((lon0 + 180.0) // 6.0) + 1
    epsg_code = (32600 if lat0 >= 0.0 else 32700) + utm_zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    x0, y0 = transformer.transform(lon0, lat0)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x) - x0, np.asarray(y) - y0


def parse_kmz_points(kmz_path: str) -> List[Dict[str, object]]:
    """Parse Placemark Point coordinates from a KMZ archive."""
    pts: List[Dict[str, object]] = []
    with zipfile.ZipFile(kmz_path, "r") as z:
        kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
        for kname in kml_names:
            root = ET.fromstring(z.read(kname))
            for pm in root.iter():
                if pm.tag.endswith("Placemark"):
                    name = ""
                    coords_text = ""
                    for child in pm.iter():
                        if child.tag.endswith("name") and not name:
                            name = (child.text or "").strip()
                        elif child.tag.endswith("Point"):
                            for c2 in child.iter():
                                if c2.tag.endswith("coordinates") and c2.text:
                                    coords_text = c2.text.strip()
                    if coords_text:
                        parts = coords_text.split()[0].split(",")
                        if len(parts) >= 2:
                            pts.append(
                                {
                                    "name": name,
                                    "lat": float(parts[1]),
                                    "lon": float(parts[0]),
                                    "alt": float(parts[2]) if len(parts) >= 3 else 0.0,
                                }
                            )
    return pts
