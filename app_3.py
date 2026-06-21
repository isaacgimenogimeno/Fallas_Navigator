"""
Carpas Fallas Navigator — Valencia
====================================
App de enrutamiento que calcula la ruta en coche más eficiente entre dos
direcciones de Valencia, esquivando activamente las calles bloqueadas por
las carpas falleras (datos abiertos del Ayuntamiento de Valencia).

Pipeline de Data Science:
  1. Ingesta GML/XML (WFS) + reproyección UTM(25830) <-> WGS84(4326)
  2. Clustering espacial no supervisado (DBSCAN) para fusionar carpas
     contiguas/duplicadas y reducir ruido geométrico.
  3. Descarga dinámica del grafo vial (OSMnx) acotado al bounding-box
     del área de estudio.
  4. Unión espacial (sjoin) carpas-buffer ∩ aristas para "bloquear" tramos.
  5. Geocodificación híbrida con fallback (Nominatim -> ArcGIS).
  6. Routing con Dijkstra/A* (NetworkX) sobre un peso compuesto:
     longitud_real + penalización fija por nodo (minimiza nº de giros).

Ejecutar:
    streamlit run app.py
"""

import time
import warnings
from typing import Optional, Set, Tuple, List

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import streamlit as st
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import ArcGIS, Nominatim
from shapely.geometry import Point, box
from sklearn.cluster import DBSCAN
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

XML_PATH = "CasalesValencia.xml"
LAYER_NAME = "Carpes_falles___Carpas_fallas"
UTM_CRS = "EPSG:32630"  # UTM zona 30N, preciso para distancias en Valencia
WGS84 = "EPSG:4326"

st.set_page_config(
    page_title="Carpas Fallas Navigator · Valencia",
    page_icon="🔥",
    layout="wide",
)

# ───────────────────────────────────────────── DATA LAYER ──────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def cargar_carpas(xml_path: str) -> gpd.GeoDataFrame:
    """Carga el WFS/GML de carpas y calcula centroides precisos en UTM."""
    gdf = gpd.read_file(xml_path, layer=LAYER_NAME)
    gdf_utm = gdf.to_crs(UTM_CRS)
    gdf_utm["centroide"] = gdf_utm.geometry.centroid
    gdf = gdf.to_crs(WGS84)
    gdf["centroide"] = gdf_utm["centroide"].to_crs(WGS84)
    gdf = gdf.rename(columns={"Id._Falla": "id_falla"})
    return gdf


@st.cache_data(show_spinner=False)
def agrupar_carpas(_gdf: gpd.GeoDataFrame, distancia_maxima: float) -> gpd.GeoDataFrame:
    """
    Clustering espacial (DBSCAN) sobre los centroides para fusionar carpas
    contiguas en un único polígono representativo (el de mayor área),
    reduciendo redundancia geométrica antes de calcular el bloqueo vial.
    """
    coords = np.array([[c.x, c.y] for c in _gdf["centroide"]])
    eps_grados = distancia_maxima / 111_320  # ≈ metros -> grados
    clusters = DBSCAN(eps=eps_grados, min_samples=1, metric="euclidean").fit_predict(coords)

    grupos = []
    for cluster_id in set(clusters):
        idxs = np.where(clusters == cluster_id)[0]
        if cluster_id == -1:
            for idx in idxs:
                grupos.append(
                    {
                        "id_falla": _gdf.iloc[idx]["id_falla"],
                        "geometry": _gdf.iloc[idx].geometry,
                        "centroide": _gdf.iloc[idx]["centroide"],
                        "num_carpas": 1,
                    }
                )
        else:
            coords_grupo = coords[idxs]
            poligonos = [_gdf.iloc[i].geometry for i in idxs]
            poligono_repr = max(poligonos, key=lambda p: p.area)
            grupos.append(
                {
                    "id_falla": f"GRUPO_{cluster_id}",
                    "geometry": poligono_repr,
                    "centroide": Point(coords_grupo.mean(axis=0)),
                    "num_carpas": len(idxs),
                }
            )
    return gpd.GeoDataFrame(grupos, crs=_gdf.crs)


@st.cache_resource(show_spinner=False)
def descargar_grafo(_gdf_carpas: gpd.GeoDataFrame, network_type: str = "drive") -> nx.MultiDiGraph:
    """Descarga el grafo vial de OSM acotado al bbox de las carpas (+margen)."""
    minx, miny, maxx, maxy = _gdf_carpas.total_bounds
    margen = 0.003  # ~300 m de margen para no aislar los extremos
    area = box(minx - margen, miny - margen, maxx + margen, maxy + margen)
    G = ox.graph_from_polygon(area, network_type=network_type, simplify=True)
    return G


@st.cache_resource(show_spinner=False)
def bloquear_aristas(
    _G: nx.MultiDiGraph, _gdf_carpas: gpd.GeoDataFrame, buffer_metros: float
) -> Tuple[nx.MultiDiGraph, int]:
    """Elimina del grafo las aristas que intersectan el buffer de seguridad de cada carpa."""
    edges = ox.graph_to_gdfs(_G, nodes=False, edges=True)
    gdf_utm = _gdf_carpas.to_crs(UTM_CRS)
    gdf_utm["geometry_buffer"] = gdf_utm.geometry.buffer(buffer_metros)
    gdf_buffer = gpd.GeoDataFrame(geometry=gdf_utm["geometry_buffer"].to_crs(WGS84), crs=WGS84)

    bloqueadas = gpd.sjoin(edges, gdf_buffer, how="inner", predicate="intersects")
    ids_bloqueadas: Set = set(bloqueadas.index)

    G_limpio = _G.copy()
    for u, v, key in list(_G.edges(keys=True)):
        if (u, v, key) in ids_bloqueadas:
            if G_limpio.has_edge(u, v, key):
                G_limpio.remove_edge(u, v, key)

    try:
        G_limpio = ox.truncate.largest_component(G_limpio, strongly=False)
    except Exception:
        pass

    return G_limpio, len(ids_bloqueadas)


def geocodificar_direccion(direccion: str) -> Tuple[Optional[float], Optional[float]]:
    """Geocodificación robusta: Nominatim (gratuito) con fallback a ArcGIS."""
    direccion_busqueda = f"{direccion}, Valencia, España"

    try:
        geolocator_nom = Nominatim(user_agent="fallas_nav_streamlit")
        geocode_nom = RateLimiter(geolocator_nom.geocode, min_delay_seconds=1)
        location = geocode_nom(
            direccion_busqueda, countrycodes="es", viewbox=[(39.3, -0.5), (39.6, -0.2)]
        )
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass

    try:
        location = ArcGIS().geocode(direccion_busqueda)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass

    try:
        location = Nominatim(user_agent="fallas_nav_streamlit_fallback").geocode(direccion)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass

    return None, None


def get_edge_length(u, v, d: dict) -> float:
    """Longitud robusta de arista; estima por geometría si falta el atributo."""
    if d.get("length") is not None:
        return d["length"]
    geom = d.get("geometry")
    if geom is not None:
        coords = list(geom.coords)
        if len(coords) >= 2:
            gdf_tmp = gpd.GeoDataFrame(geometry=[geom], crs=WGS84).to_crs(UTM_CRS)
            return gdf_tmp.geometry.length.iloc[0]
    return 50.0


def calcular_ruta_penalizada(
    G: nx.MultiDiGraph,
    lat_origen: float,
    lon_origen: float,
    lat_dest: float,
    lon_dest: float,
    penalty_nodo: int = 100,
) -> Tuple[Optional[List[Tuple[float, float]]], Optional[float], Optional[float], Optional[int]]:
    """Ruta más corta penalizando intersecciones (minimiza nº de nodos/giros)."""
    try:
        origen = ox.distance.nearest_nodes(G, X=lon_origen, Y=lat_origen)
        destino = ox.distance.nearest_nodes(G, X=lon_dest, Y=lat_dest)
    except Exception as e:
        raise RuntimeError(f"No se pudo localizar un nodo cercano: {e}")

    try:
        ruta = nx.shortest_path(
            G, origen, destino, weight=lambda u, v, d: get_edge_length(u, v, d) + penalty_nodo
        )
    except nx.NetworkXNoPath:
        return None, None, None, None
    except nx.NodeNotFound:
        return None, None, None, None

    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in ruta]

    distancia = 0.0
    for i in range(len(ruta) - 1):
        u, v = ruta[i], ruta[i + 1]
        edge_dict = G.get_edge_data(u, v)
        if edge_dict:
            datos = edge_dict[next(iter(edge_dict))]
            distancia += get_edge_length(u, v, datos)

    coste_total = distancia + penalty_nodo * len(ruta)
    return coords, distancia, coste_total, len(ruta)


def carpas_cercanas_a_ruta(
    gdf_carpas: gpd.GeoDataFrame, ruta_coords: List[Tuple[float, float]], buffer_m: float = 150
) -> gpd.GeoDataFrame:
    """Filtra las carpas próximas a la ruta calculada (para no saturar el mapa)."""
    from shapely.geometry import LineString

    ruta_line = LineString([(lon, lat) for lat, lon in ruta_coords])
    gdf_ruta = gpd.GeoDataFrame(geometry=[ruta_line], crs=WGS84).to_crs(UTM_CRS)
    gdf_ruta["geometry"] = gdf_ruta.geometry.buffer(buffer_m)
    gdf_ruta_buffer = gdf_ruta.to_crs(WGS84)
    return gpd.sjoin(gdf_carpas, gdf_ruta_buffer, how="inner", predicate="intersects")


def crear_mapa(
    gdf_carpas: gpd.GeoDataFrame,
    ruta_coords: Optional[List[Tuple[float, float]]] = None,
    origen: Optional[Tuple[float, float]] = None,
    destino: Optional[Tuple[float, float]] = None,
    distancia: Optional[float] = None,
) -> folium.Map:
    """Construye un mapa Folium limpio: ruta, marcadores y carpas cercanas (sin sombreado de buffer)."""
    if ruta_coords:
        lats = [p[0] for p in ruta_coords]
        lons = [p[1] for p in ruta_coords]
        centro = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]
        zoom = 15
    else:
        centro = [39.4699, -0.3763]  # Centro de Valencia
        zoom = 13

    m = folium.Map(location=centro, zoom_start=zoom, tiles="cartodbpositron")

    carpas_mostrar = gdf_carpas
    if ruta_coords:
        try:
            carpas_mostrar = carpas_cercanas_a_ruta(gdf_carpas, ruta_coords)
        except Exception:
            carpas_mostrar = gdf_carpas

    for _, row in carpas_mostrar.iterrows():
        cent = row["centroide"]
        folium.CircleMarker(
            location=(cent.y, cent.x),
            radius=7,
            color="#d62728",
            fill=True,
            fill_color="#d62728",
            fill_opacity=0.55,
            weight=1.5,
            popup=folium.Popup(
                f"🔥 Falla {row['id_falla']}<br>Carpas fusionadas: {row.get('num_carpas', 1)}",
                max_width=200,
            ),
            tooltip=f"Falla {row['id_falla']}",
        ).add_to(m)

    if origen:
        folium.Marker(
            location=origen,
            icon=folium.Icon(color="green", icon="play"),
            popup="📍 Origen",
            tooltip="Origen",
        ).add_to(m)

    if destino:
        folium.Marker(
            location=destino,
            icon=folium.Icon(color="orange", icon="flag"),
            popup="🏁 Destino",
            tooltip="Destino",
        ).add_to(m)

    if ruta_coords:
        folium.PolyLine(
            locations=ruta_coords,
            color="#1f77b4",
            weight=5,
            opacity=0.9,
            tooltip=f"Distancia: {distancia:.0f} m" if distancia else "Ruta",
        ).add_to(m)

    return m


# ───────────────────────────────────────────── UI LAYER ──────────────────────────────────────────────


def main() -> None:
    st.title("🔥 Carpas Fallas Navigator — Valencia")
    st.caption(
        "Calcula la ruta en coche más eficiente evitando las calles cortadas por las carpas falleras."
    )

    with st.sidebar:
        st.header("⚙️ Parámetros del modelo")
        distancia_cluster = st.slider(
            "Distancia de fusión de carpas (m) — DBSCAN ε", 5, 60, 30, 5,
            help="Carpas separadas menos que este umbral se fusionan en un único grupo.",
        )
        buffer_bloqueo = st.slider(
            "Buffer de seguridad alrededor de cada carpa (m)", 1, 20, 5, 1,
            help="Margen de seguridad usado para bloquear los tramos de calle que la carpa invade.",
        )
        penalty_nodo = st.slider(
            "Penalización por intersección (m equivalentes)", 0, 300, 100, 10,
            help="Penaliza el paso por nodos/cruces; valores altos = rutas con menos giros.",
        )
        st.divider()
        st.caption("Fuente: Geoportal Ayuntamiento de Valencia (WFS) + OpenStreetMap (OSMnx).")

    col_a, col_b = st.columns(2)
    with col_a:
        origen_txt = st.text_input("📍 Dirección de origen", placeholder="Ej. Calle Colón, 15")
    with col_b:
        destino_txt = st.text_input("🏁 Dirección de destino", placeholder="Ej. Plaza del Ayuntamiento")

    calcular = st.button("🔄 Calcular ruta", type="primary", use_container_width=True)

    status_box = st.empty()
    stats_box = st.empty()

    # `mapa_actual` se construye UNA sola vez por rerun y se renderiza con una
    # única key estable ("mapa_principal"). Así evitamos que Streamlit cree
    # varios componentes folium distintos en la misma sesión (causa del
    # parpadeo / desaparición intermitente del mapa).
    if "mapa_actual" not in st.session_state:
        try:
            gdf_inicial = cargar_carpas(XML_PATH)
            st.session_state["mapa_actual"] = crear_mapa(gdf_inicial)
        except Exception as e:
            status_box.warning(f"No se pudo precargar el dataset de carpas: {e}")
            st.session_state["mapa_actual"] = folium.Map(
                location=[39.4699, -0.3763], zoom_start=13, tiles="cartodbpositron"
            )

    def ejecutar_calculo() -> None:
        """Usa `return` (no `st.stop()`) para no impedir el render final del mapa."""
        if not origen_txt.strip() or not destino_txt.strip():
            status_box.error("❌ Debes introducir tanto el origen como el destino.")
            return

        try:
            with st.spinner("📦 Cargando y agrupando carpas falleras…"):
                gdf_carpas_raw = cargar_carpas(XML_PATH)
                gdf_carpas = agrupar_carpas(gdf_carpas_raw, distancia_cluster)

            with st.spinner("🗺️ Descargando red viaria de OpenStreetMap…"):
                G_original = descargar_grafo(gdf_carpas)

            with st.spinner("🚧 Bloqueando tramos invadidos por carpas…"):
                G_limpio, n_bloqueadas = bloquear_aristas(G_original, gdf_carpas, buffer_bloqueo)

            with st.spinner("📍 Geocodificando direcciones…"):
                lat1, lon1 = geocodificar_direccion(origen_txt)
                lat2, lon2 = geocodificar_direccion(destino_txt)

            if None in (lat1, lon1):
                status_box.error(f"❌ No se pudo geocodificar el origen: '{origen_txt}'.")
                return
            if None in (lat2, lon2):
                status_box.error(f"❌ No se pudo geocodificar el destino: '{destino_txt}'.")
                return

            bbox = gdf_carpas.total_bounds
            if not (bbox[0] <= lon1 <= bbox[2] and bbox[1] <= lat1 <= bbox[3]):
                status_box.warning("⚠️ El origen está fuera del área cubierta por las carpas registradas.")
            if not (bbox[0] <= lon2 <= bbox[2] and bbox[1] <= lat2 <= bbox[3]):
                status_box.warning("⚠️ El destino está fuera del área cubierta por las carpas registradas.")

            with st.spinner("🧭 Calculando ruta óptima (Dijkstra ponderado)…"):
                t0 = time.time()
                try:
                    coords, distancia, coste, num_nodos = calcular_ruta_penalizada(
                        G_limpio, lat1, lon1, lat2, lon2, penalty_nodo=penalty_nodo
                    )
                except RuntimeError as e:
                    status_box.error(f"❌ {e}")
                    return
                t_calc = time.time() - t0

            if coords is None:
                status_box.error(
                    "🚫 No existe una ruta disponible: el origen y/o destino quedan completamente "
                    "aislados por los cortes de calle de las carpas falleras con la configuración actual. "
                    "Prueba a reducir el buffer de seguridad o revisa las direcciones introducidas."
                )
                st.session_state["mapa_actual"] = crear_mapa(
                    gdf_carpas, origen=(lat1, lon1), destino=(lat2, lon2)
                )
                return

            status_box.success("✅ Ruta calculada con éxito.")

            st.session_state["mapa_actual"] = crear_mapa(
                gdf_carpas,
                ruta_coords=coords,
                origen=(lat1, lon1),
                destino=(lat2, lon2),
                distancia=distancia,
            )

            with stats_box.container():
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📏 Distancia", f"{distancia/1000:.2f} km")
                c2.metric("🚦 Nodos/giros", f"{num_nodos}")
                c3.metric("🚧 Calles bloqueadas", f"{n_bloqueadas}")
                c4.metric("⏱️ Tiempo cálculo", f"{t_calc:.2f} s")

        except FileNotFoundError:
            status_box.error(
                f"❌ No se encontró el fichero '{XML_PATH}'. Colócalo en el directorio de la app."
            )
        except Exception as e:
            status_box.error(f"❌ Error inesperado durante el procesamiento: {e}")

    if calcular:
        ejecutar_calculo()

    # Render único del mapa por cada ejecución del script, con key estable.
    # `returned_objects=[]` evita que las interacciones del usuario con el mapa
    # (pan/zoom/click) disparen un reenvío de datos al servidor que provoque
    # un rerun en bucle (causa típica de parpadeo en st_folium).
    st_folium(
        st.session_state["mapa_actual"],
        width=None,
        height=560,
        key="mapa_principal",
        returned_objects=[],
    )


if __name__ == "__main__":
    main()
