from detector import DetectionApp
import cv2
import base64
import logging
import time
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import Flask, Response, url_for


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] (%(threadName)-9s) %(message)s')


config = {
    'display': False,
    'speak': False,
    'camera_device_id': 0,
    'faces': {
        'detect': True,
        'shrink_frames': True,
        'anchor_images_path': './images'
    },
    'objects': {
        'detect': False,
        'shrink_frames': False
    }
}

detector = DetectionApp(config=config)


def frame_generator():
    for data in detector.detection_generator(paint_data_on_frame=True):
        logging.debug("generator reading frame")
        frame, faces_data, objects_data, fps_data = data
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')


def bulma_field(label, component):
    """
    Handle boiler plate stuff for putting a label on a dcc / input field
    """
    return html.Div(className='field', children=[
        html.Label(className='label', children=label),
        html.Div(className='control', children=[component])
    ])


def bulma_checkbox(id, label):
    return html.Div(className="field", children=[
        html.Div(className="control", children=[
            html.Label(className="checkbox", children=[
                dcc.Input(id=id, type='checkbox'),
                html.Span(label)
            ])
        ])
    ])


app = dash.Dash()

app.css.append_css({'external_url': 'https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.1/css/bulma.min.css'})
app.css.append_css({'external_url': 'https://use.fontawesome.com/releases/v5.4.1/css/all.css'})

app.layout = html.Div(className='container is-fluid', children=[
    html.Div(className='columns', children=[
        html.Div(className='column', children=[
            html.H1(children='Data Science Lab AI Cam'),

            dcc.Input(id='dummy', type='hidden'),

            html.Div(className='control', children=[
                html.Button(id='btn-start-stop', className='button is-info', n_clicks=0, children='Start')
            ]),
            bulma_field(
                label='Detection modes',
                component=dcc.Checklist(id='input-detect-mode', options=[
                    {'value': 'faces', 'label': 'Face recognition'},
                    {'value': 'objects', 'label': 'Object recognition'},
                ], values=['faces'], labelStyle={'display': 'block'})
            ),
            bulma_field(
                label='Camera source',
                component=dcc.RadioItems(id='input-camera', options=[
                    {'value': 'onboard', 'label': 'On-board webcam'},
                    {'value': 'network', 'label': 'Network IP camera'},
                ], value='onboard', labelStyle={'display': 'block'})
            )
        ]),
        html.Div(className='column is-three-quarters', children=[
            html.Div(className="column", children=[
                html.Img(id="videofeed", src='/static/frame.png')
            ])
        ])
    ])
])


# @app.callback(
#     Output('dummy', 'value'),
#     [Input('btn-stop', 'n_clicks')]
# )
# def stop(n_clicks):
#     if n_clicks == 0:
#         pass
#     elif n_clicks % 2 == 0:
#         detector.reset()
#     else:
#         detector.stop()
#     raise PreventUpdate()


@app.callback(
    Output('btn-start-stop', 'children'),
    [Input('btn-start-stop', 'n_clicks')]
)
def play_pause_button(n_clicks):
    if n_clicks % 2 == 0:
        return html.Span(className="icon", children=[
            html.I(className='fas fa-play')
        ])
    else:
        return html.Span(className="icon", children=[
            html.I(className='fas fa-pause')
        ])


@app.callback(
    Output('videofeed', 'src'),
    [Input('btn-start-stop', 'n_clicks')],
    [State('input-detect-mode', 'values'),
     State('input-camera', 'value')]
)
def start_stop(n_clicks, detect_mode_values, camera_source):
    if n_clicks == 0:
        return '/static/frame.png'
    elif n_clicks % 2 == 0:
        logging.info("STOP")
        detector.stop()
        return '/static/frame.png'
    else:
        logging.info("START")
        # handle detection mode settings
        detector.faces = 'faces' in detect_mode_values
        detector.objects = 'objects' in detect_mode_values

        # handle video source settings
        if camera_source != detector.config['camera_device_id']:
            detector.config['camera_device_id'] = ('network' if camera_source == 'network' else 0)
            detector.set_video_source()

        detector.reset()
        return '/video_feed'


@app.server.route('/video_feed')
def video_feed():
    return Response(frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame'
                    )


if __name__ == '__main__':
    app.run_server(debug=True,
                   port=8234,
                   use_reloader=False
                   )
