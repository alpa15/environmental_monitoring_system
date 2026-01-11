# dashboard/pages/change_detection.py
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional, Tuple, List

import dash
from dash import html, dcc, Input, Output, State, callback, no_update

from eo.ee.pipeline import BBox, run_single_year_analysis, run_two_year_comparison  # [file:3]

dash.register_page(__name__, path="/", name="Change Detection")  # [web:6]

OUTPUT_DIR_DEFAULT = Path("src") / "eo" / "data" / "ee" / "outputs"

IMAGES_STYLE_DEFAULT = {
    "display": "grid",
    "gridTemplateColumns": "1fr",  # full width + vertical stacking
    "gap": "12px",
    "alignItems": "start",
}


def _validate_bbox(min_lat: float, max_lat: float, min_lon: float, max_lon: float) -> Tuple[bool, str]:
    if min_lat is None or max_lat is None or min_lon is None or max_lon is None:
        return False, "Bounding box is incomplete."
    if min_lat >= max_lat:
        return False, "Latitude: min must be < max."
    if min_lon >= max_lon:
        return False, "Longitude: min must be < max."
    if not (-90.0 <= float(min_lat) <= 90.0 and -90.0 <= float(max_lat) <= 90.0):
        return False, "Latitude out of range (-90, 90)."
    if not (-180.0 <= float(min_lon) <= 180.0 and -180.0 <= float(max_lon) <= 180.0):
        return False, "Longitude out of range (-180, 180)."
    return True, ""


def _img_to_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _expected_outputs_single(out_dir: Path, year: int) -> List[Path]:
    return [out_dir / f"single_year_{year}.png"]  # [file:3]


def _expected_outputs_compare(out_dir: Path, year_a: int, year_b: int) -> List[Path]:
    return [
        out_dir / f"comparison_{year_a}_{year_b}.png",
        out_dir / f"delta_{year_a}_{year_b}.png",
    ]  # [file:3]


def _image_card(path: Path) -> html.Figure:
    uri = _img_to_data_uri(path)
    if uri is None:
        return html.Figure([html.Div(f"Missing file: {path}", style={"color": "crimson"})])

    return html.Figure(
        children=[
            html.Img(src=uri, style={"width": "100%", "border": "1px solid #e5e7eb"}),
            html.Figcaption(path.name, style={"fontSize": "12px", "color": "#374151"}),
        ]
    )


layout = html.Div(
    style={"padding": "16px"},
    children=[
        html.H2("Change Detection (GEE)"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "360px 1fr", "gap": "16px", "alignItems": "start"},
            children=[
                # Controls
                html.Div(
                    style={"border": "1px solid #e5e7eb", "borderRadius": "8px", "padding": "12px"},
                    children=[
                        html.H4("Parameters"),
                        html.Label("Min latitude"),
                        dcc.Input(id="cd-min-lat", type="number", value=41.85),
                        html.Br(),
                        html.Label("Max latitude"),
                        dcc.Input(id="cd-max-lat", type="number", value=41.95),
                        html.Br(),
                        html.Label("Min longitude"),
                        dcc.Input(id="cd-min-lon", type="number", value=12.45),
                        html.Br(),
                        html.Label("Max longitude"),
                        dcc.Input(id="cd-max-lon", type="number", value=12.55),
                        html.Hr(),
                        html.Label("Start year"),
                        dcc.Input(id="cd-year-a", type="number", min=2015, max=2026, step=1, value=2018),
                        html.Br(),
                        dcc.Checklist(
                            id="cd-enable-compare",
                            options=[{"label": "Compare with end year", "value": "on"}],
                            value=["on"],
                        ),
                        html.Label("End year"),
                        dcc.Input(id="cd-year-b", type="number", min=2015, max=2026, step=1, value=2024),
                        html.Hr(),
                        html.Label("Season"),
                        dcc.Dropdown(
                            id="cd-season",
                            options=[
                                {"label": "spring", "value": "spring"},
                                {"label": "summer", "value": "summer"},
                                {"label": "autumn", "value": "autumn"},
                                {"label": "winter", "value": "winter"},
                            ],
                            value="summer",
                            clearable=False,
                        ),
                        html.Br(),
                        html.Label("Max cloud cover (%)"),
                        dcc.Slider(id="cd-cloud", min=0, max=100, step=5, value=50, tooltip={"placement": "bottom"}),
                        html.Br(),
                        html.Label("Output directory"),
                        dcc.Input(id="cd-outdir", type="text", value=str(OUTPUT_DIR_DEFAULT), style={"width": "100%"}),
                        html.Br(),
                        html.Br(),
                        html.Button("Run", id="cd-run", n_clicks=0),
                        html.Div(id="cd-status", style={"marginTop": "10px"}),
                    ],
                ),
                # Outputs (wrapped in Loading)
                html.Div(
                    children=[
                        html.H4("Outputs"),
                        dcc.Loading(
                            id="cd-loading",
                            type="default",
                            children=html.Div(id="cd-images", style=IMAGES_STYLE_DEFAULT),
                        ),  # dcc.Loading spinner [web:89]
                    ]
                ),
            ],
        ),
    ],
)


@callback(
    Output("cd-year-b", "disabled"),
    Input("cd-enable-compare", "value"),
)
def toggle_end_year(enable_value):
    enabled = bool(enable_value) and ("on" in enable_value)
    return not enabled


# 1) Fast callback: immediately clear previous results and status when Run is clicked
@callback(
    Output("cd-status", "children", allow_duplicate=True),
    Output("cd-images", "children", allow_duplicate=True),
    Input("cd-run", "n_clicks"),
    prevent_initial_call=True,
)
def clear_on_run(n_clicks):
    # This removes "Done." and clears images right away.
    # The dcc.Loading spinner will show while the heavy callback runs. [web:89]
    return "", []


# 2) Heavy callback: runs the pipeline and then renders new images
@callback(
    Output("cd-status", "children"),
    Output("cd-images", "children"),
    Output("cd-images", "style"),
    Input("cd-run", "n_clicks"),
    State("cd-min-lat", "value"),
    State("cd-max-lat", "value"),
    State("cd-min-lon", "value"),
    State("cd-max-lon", "value"),
    State("cd-year-a", "value"),
    State("cd-year-b", "value"),
    State("cd-enable-compare", "value"),
    State("cd-season", "value"),
    State("cd-cloud", "value"),
    State("cd-outdir", "value"),
    prevent_initial_call=True,
)
def run_change_detection(
    n_clicks,
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    year_a,
    year_b,
    enable_compare,
    season,
    cloud_cover,
    out_dir,
):
    ok, msg = _validate_bbox(min_lat, max_lat, min_lon, max_lon)
    if not ok:
        return html.Div(msg, style={"color": "crimson"}), [], IMAGES_STYLE_DEFAULT

    if year_a is None:
        return html.Div("Start year is missing.", style={"color": "crimson"}), [], IMAGES_STYLE_DEFAULT

    do_compare = bool(enable_compare) and ("on" in enable_compare) and (year_b is not None) and (int(year_b) != int(year_a))

    bbox = BBox(
        min_lon=float(min_lon),
        min_lat=float(min_lat),
        max_lon=float(max_lon),
        max_lat=float(max_lat),
    )
    out_path = Path(out_dir) if out_dir else OUTPUT_DIR_DEFAULT

    try:
        if not do_compare:
            run_single_year_analysis(
                bbox=bbox,
                year=int(year_a),
                season=str(season),
                cloud_cover=int(cloud_cover),
                out_dir=out_path,
                show_plot=False,
                save_plot=True,
                export_drive=False,
            )  # [file:3]
            outputs = _expected_outputs_single(out_path, int(year_a))
        else:
            run_two_year_comparison(
                bbox=bbox,
                year_a=int(year_a),
                year_b=int(year_b),
                season=str(season),
                cloud_cover=int(cloud_cover),
                out_dir=out_path,
                show_plot=False,
                save_plot=True,
            )  # [file:3]
            outputs = _expected_outputs_compare(out_path, int(year_a), int(year_b))

        cards = [_image_card(p) for p in outputs]
        return html.Div("Done.", style={"color": "green"}), cards, IMAGES_STYLE_DEFAULT

    except Exception as e:
        return html.Pre(str(e), style={"color": "crimson", "whiteSpace": "pre-wrap"}), [], IMAGES_STYLE_DEFAULT
