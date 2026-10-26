import os
import streamlit.components.v1 as components

_component = components.declare_component(
    "interview_audio_player",
    path=os.path.dirname(__file__),
)

def audio_player(audio_b64, question_key, autoplay=True, key=None, default=False):
    return _component(
        audio_b64=audio_b64,
        question_key=question_key,
        autoplay=autoplay,
        key=key,
        default=default,
    )
