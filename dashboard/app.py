# dashboard/app.py
from __future__ import annotations

import dash
from dash import Dash, html, dcc

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)  # [web:6]
server = app.server

app.layout = html.Div(
    children=[
        dcc.Location(id="url", refresh=False),
        html.Div(
            style={
                "display": "flex",
                "gap": "12px",
                "alignItems": "baseline",
                "padding": "12px 16px",
                "borderBottom": "1px solid #e5e7eb",
            },
            children=[
                html.H3("EO Dashboard", style={"margin": 0}),
                dcc.Link("Change Detection", href="/", style={"marginLeft": "12px"}),
            ],
        ),
        html.Div(style={"padding": "8px 16px"}, children=[dash.page_container]),
    ]
)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050, use_reloader=False)  # [web:89]
