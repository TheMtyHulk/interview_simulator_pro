import os
import json
import base64
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from typing_extensions import TypedDict
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")

st.set_page_config(page_title="Interview Studio", page_icon="✦", layout="wide")

if not API_KEY:
    st.error("Add GEMINI_API_KEY to your .env file, then restart Streamlit.")
    st.stop()

client = genai.Client(api_key=API_KEY)
llm = ChatGoogleGenerativeAI(
    model=MODEL, google_api_key=API_KEY, temperature=0.2, max_retries=1
)

# Local custom component: reports the real browser audio 'ended' event.
_audio_component = components.declare_component(
    "interview_audio_player",
    path=os.path.join(os.path.dirname(__file__), "audio_player"),
)

def as_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("text"):
                out.append(part["text"])
        return "\n".join(out)
    return str(content)

def ask_llm(prompt):
    return as_text(llm.invoke(prompt).content).strip()

def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Gemini did not return valid JSON.")
    return json.loads(text[start:end + 1])

def clean_list(value):
    return [str(x) for x in value] if isinstance(value, list) else []

def clamp_score(value):
    try:
        return max(0, min(10, int(float(value))))
    except (TypeError, ValueError):
        return 0

@st.cache_data(show_spinner=False, ttl=3600, max_entries=100)
def generate_speech(question):
    response = client.interactions.create(
        model=TTS_MODEL,
        input=[{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": question,
                "annotations": [{
                    "type": "speech_metadata",
                    "style": "Warm, natural, clear, confident, professional interviewer. Conversational pace."
                }]
            }]
        }],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": "Kore"}]},
    )
    data = response.output_audio.data
    return base64.b64decode(data) if isinstance(data, str) else data

def transcribe_audio(audio_file):
    result = client.models.generate_content(
        model=MODEL,
        contents=[
            "Transcribe the candidate's speech accurately. Return only the transcript.",
            types.Part.from_bytes(
                data=audio_file.getvalue(),
                mime_type=audio_file.type or "audio/wav",
            ),
        ],
    )
    text = result.text or ""
    if not text.strip():
        raise ValueError("No transcript detected. Please record again.")
    return text.strip()

class InterviewState(TypedDict):
    role: str
    level: str
    max_questions: int
    question_number: int
    question: str
    answer: str
    history: list
    evaluation: dict
    final_report: str
    finished: bool

def evaluate_answer(state: InterviewState):
    next_number = state["question_number"] + 1
    hard_limit_reached = next_number >= state["max_questions"]
    prompt = f"""
You are a fair, constructive professional interviewer.
Role: {state['role']}
Level: {state['level']}
Current question: {state['question']}
Candidate answer: {state['answer']}
Previous turns: {json.dumps(state['history'], ensure_ascii=False)}
This is answer {next_number}. The absolute maximum is {state['max_questions']} answers.
Hard limit reached after this answer: {hard_limit_reached}

You decide whether the interview is complete based on the candidate's performance,
the role, coverage of useful topics, and whether further questions would add value.
Usually conduct several questions, but end earlier if the interview has adequately
assessed the candidate or the conversation naturally reaches a conclusion.
NEVER continue beyond the hard limit of {state['max_questions']} answers.

Return ONLY JSON:
{{
 "score": 0,
 "strengths": ["specific strength"],
 "improvements": ["specific improvement"],
 "feedback": "concise and actionable feedback",
 "ideal_answer_points": ["point a strong answer should cover"],
 "finished": false,
 "next_question": "one adaptive question or empty string if finished"
}}
Score from 0 to 10. Be fair to the experience level. Do not invent facts.
Set "finished" to true when the interview should end. If finished, next_question
must be an empty string. If not finished, use the answer and history to ask a relevant
follow-up or next question. Explore gaps or interesting points when useful; avoid repetition.
{"You MUST set finished=true and next_question to an empty string because the hard limit has been reached." if hard_limit_reached else "Decide whether to continue or finish based on the interview so far."}
"""
    data = parse_json(ask_llm(prompt))
    finished = hard_limit_reached or bool(data.get("finished", False))
    evaluation = {
        "score": clamp_score(data.get("score", 0)),
        "strengths": clean_list(data.get("strengths")),
        "improvements": clean_list(data.get("improvements")),
        "feedback": str(data.get("feedback", "No feedback returned.")),
        "ideal_answer_points": clean_list(data.get("ideal_answer_points")),
    }
    next_question = str(data.get("next_question", "")).strip()
    if not finished and not next_question:
        next_question = f"Can you walk me through a practical {state['role']} problem and how you would solve it?"
    if finished:
        next_question = ""
    turn = {"question": state["question"], "answer": state["answer"], "evaluation": evaluation}
    return {
        "evaluation": evaluation,
        "history": state["history"] + [turn],
        "question_number": next_number,
        "question": next_question,
        "finished": finished,
    }

def make_report(state: InterviewState):
    turns = state["history"]
    scores = [t["evaluation"]["score"] for t in turns]
    avg = sum(scores) / len(scores) if scores else 0
    strengths, improvements = [], []
    for t in turns:
        strengths += t["evaluation"]["strengths"]
        improvements += t["evaluation"]["improvements"]
    strengths = list(dict.fromkeys(strengths))
    improvements = list(dict.fromkeys(improvements))
    assessment = "Strong performance" if avg >= 8 else "Good foundation with room to improve" if avg >= 6 else "Developing; more practice recommended" if avg >= 4 else "More preparation recommended"
    report = f"""## Interview summary

- **Role:** {state['role']}
- **Level:** {state['level']}
- **Questions answered:** {len(turns)}
- **Average score:** {avg:.1f}/10
- **Assessment:** {assessment}

### Strengths
{chr(10).join('- ' + x for x in strengths[:8]) or '- Continue practising to identify strengths.'}

### Areas to improve
{chr(10).join('- ' + x for x in improvements[:8]) or '- Continue practising role-specific questions.'}

### Suggested next steps
1. Review the improvement points above.
2. Practise explaining reasoning with concrete examples.
3. Repeat a mock interview and compare results.

*This is AI-generated practice feedback, not a hiring decision.*
"""
    return {"final_report": report, "finished": True}

def route(state):
    return "report" if state["finished"] else "end"

graph_builder = StateGraph(InterviewState)
graph_builder.add_node("evaluate_answer", evaluate_answer)
graph_builder.add_node("final_report", make_report)
graph_builder.add_edge(START, "evaluate_answer")
graph_builder.add_conditional_edges("evaluate_answer", route, {"report": "final_report", "end": END})
graph_builder.add_edge("final_report", END)
graph = graph_builder.compile()

DEFAULTS = {
    "started": False, "finished": False, "role": "Python Developer",
    "level": "Fresher", "max_questions": 5, "question_number": 0,
    "question": "", "history": [], "evaluation": {}, "final_report": "",
    "audio_ready": False, "audio_bytes": None, "audio_question": "",
    "audio_error": "", "last_transcript": "",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

def reset_interview():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value

def first_question(role, level):
    return ask_llm(f"""You are a professional interviewer. Role: {role}. Level: {level}.
Generate one concise opening interview question. Return only the question.""")

def run_answer(answer):
    state = {
        "role": st.session_state.role,
        "level": st.session_state.level,
        "max_questions": st.session_state.max_questions,
        "question_number": st.session_state.question_number,
        "question": st.session_state.question,
        "answer": answer,
        "history": st.session_state.history,
        "evaluation": st.session_state.evaluation,
        "final_report": st.session_state.final_report,
        "finished": False,
    }
    result = graph.invoke(state)
    for key in ("history", "question_number", "question", "evaluation", "finished", "final_report"):
        st.session_state[key] = result[key]
    st.session_state.audio_ready = False
    st.session_state.audio_bytes = None
    st.session_state.audio_question = ""
    st.session_state.audio_error = ""

# ---------- Polished visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#182230; --muted:#687386; --line:#e6eaf0; --brand:#4f46e5; --soft:#f5f7ff; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f8faff 0%,#ffffff 280px); color:var(--ink); }
.block-container { max-width:1180px; padding-top:1.6rem; padding-bottom:3rem; }
.hero { padding:28px 30px; border:1px solid #e4e8f7; border-radius:24px; background:linear-gradient(120deg,#fff 0%,#f1f3ff 100%); box-shadow:0 12px 35px rgba(32,45,90,.05); margin-bottom:22px; }
.eyebrow { color:#5145cd; font-size:.75rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
.hero h1 { font-family:'Manrope',sans-serif; letter-spacing:-.04em; font-size:2.1rem; line-height:1.15; margin:.5rem 0; color:#171d38; }
.hero p { color:#687386; margin:0; font-size:1rem; }
.panel { background:white; border:1px solid var(--line); border-radius:20px; padding:22px; box-shadow:0 8px 24px rgba(20,35,70,.035); margin-bottom:16px; }
.question-card { background:#f6f7ff; border:1px solid #e3e6ff; border-radius:18px; padding:22px; margin:14px 0 18px; }
.question-label { color:#5952ce; font-size:.75rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
.question-text { color:#1d2540; font-family:'Manrope',sans-serif; font-size:1.35rem; line-height:1.5; font-weight:700; margin-top:8px; }
.status { padding:12px 15px; border-radius:12px; background:#f1f5f9; color:#465366; font-size:.9rem; margin:8px 0 14px; }
.status.ready { background:#eaf8f0; color:#17623c; }
.small-muted { color:#7a8495; font-size:.88rem; }
.session-stats { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; margin:18px 0 22px; width:100%; }
.stat-card { min-width:0; overflow:hidden; box-sizing:border-box; background:#fff; border:1px solid #e6eaf0; border-radius:16px; padding:16px 18px; box-shadow:0 6px 18px rgba(20,35,70,.035); }
.stat-label { font-size:.69rem; font-weight:800; letter-spacing:.12em; color:#7a8495; margin-bottom:9px; }
.stat-value { font-family:'Manrope',sans-serif; font-size:clamp(1.15rem,2vw,1.75rem); line-height:1.25; font-weight:800; color:#1d2540; overflow-wrap:anywhere; }
.stat-value span { font-family:'DM Sans',sans-serif; font-size:.82rem; font-weight:600; color:#8992a2; }
.stat-text { font-size:clamp(1rem,1.6vw,1.35rem); }
.stat-note { margin-top:7px; font-size:.78rem; color:#8992a2; }
@media (max-width:700px) { .session-stats { grid-template-columns:1fr; gap:9px; } .stat-card { padding:13px 15px; } .stat-value { font-size:1.25rem; } }
div[data-testid="stSidebar"] { background:#fbfcff; border-right:1px solid #edf0f5; }
.stButton>button, .stDownloadButton>button { border-radius:12px; min-height:2.8rem; font-weight:700; }
.stButton>button[kind="primary"] { background:#4f46e5; border-color:#4f46e5; color:#ffffff !important; }

/* Keep typed answers readable regardless of the user's Streamlit theme. */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background-color:#ffffff !important;
    color:#182230 !important;
    -webkit-text-fill-color:#182230 !important;
    caret-color:#4f46e5 !important;
    border:1px solid #cbd5e1 !important;
    border-radius:12px !important;
}
div[data-testid="stTextArea"] textarea::placeholder,
div[data-testid="stTextInput"] input::placeholder {
    color:#778196 !important;
    -webkit-text-fill-color:#778196 !important;
    opacity:1 !important;
}
div[data-testid="stTextArea"] textarea:focus,
div[data-testid="stTextInput"] input:focus {
    border-color:#4f46e5 !important;
    box-shadow:0 0 0 1px #4f46e5 !important;
}

/* Explicit colors for JSON download button in normal, hover and focus states. */
div[data-testid="stDownloadButton"] > button,
div[data-testid="stDownloadButton"] > button:link,
div[data-testid="stDownloadButton"] > button:visited {
    background:#4f46e5 !important;
    background-color:#4f46e5 !important;
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    border:1px solid #4f46e5 !important;
    border-radius:12px !important;
    font-weight:700 !important;
    transition:background-color .15s ease, border-color .15s ease !important;
}
div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:focus,
div[data-testid="stDownloadButton"] > button:active {
    background:#3730a3 !important;
    background-color:#3730a3 !important;
    color:#ffffff !important;
    -webkit-text-fill-color:#ffffff !important;
    border-color:#3730a3 !important;
}
div[data-testid="stDownloadButton"] > button p,
div[data-testid="stDownloadButton"] > button span,
div[data-testid="stDownloadButton"] > button svg {
    color:#ffffff !important;
    fill:#ffffff !important;
}
/* High-contrast Streamlit metrics, including the per-answer Score / 10. */
div[data-testid="stMetric"] {
    border:1px solid #dfe5ef !important;
    padding:16px 18px !important;
    border-radius:16px !important;
    background:#ffffff !important;
    color:#172033 !important;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color:#596579 !important;
    -webkit-text-fill-color:#596579 !important;
    opacity:1 !important;
}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div,
div[data-testid="stMetricValue"] > div {
    color:#2636a8 !important;
    -webkit-text-fill-color:#2636a8 !important;
    font-weight:800 !important;
}
.answer-score-card {
    box-sizing:border-box;
    min-width:0;
    padding:18px 20px;
    background:#ffffff !important;
    border:1px solid #dfe5ef;
    border-radius:16px;
    box-shadow:0 4px 16px rgba(20,35,70,.06);
    color:#172033 !important;
}
.answer-score-label { color:#596579 !important; font-size:12px; font-weight:800; letter-spacing:1.1px; margin-bottom:8px; }
.answer-score-value { color:#2636a8 !important; -webkit-text-fill-color:#2636a8 !important; font-size:clamp(30px,4vw,40px); line-height:1.2; font-weight:800; }
.answer-score-value span { color:#667085 !important; -webkit-text-fill-color:#667085 !important; font-size:17px; font-weight:600; }
.score-summary { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; width:100%; margin:18px 0 24px; }
.score-card { box-sizing:border-box; min-width:0; padding:24px; background:#fff; border:1px solid #dfe5ef; border-radius:16px; box-shadow:0 4px 16px rgba(20,35,70,.06); color:#172033 !important; overflow-wrap:anywhere; }
.score-label { color:#596579 !important; font-size:12px; font-weight:800; letter-spacing:1.2px; margin-bottom:12px; }
.score-number { color:#2636a8 !important; font-size:clamp(30px,5vw,44px); line-height:1.2; font-weight:800; }
.score-number span { color:#667085 !important; font-size:18px; font-weight:600; margin-left:5px; }
.score-caption { color:#667085 !important; font-size:13px; margin-top:10px; }
@media (max-width:600px) { .score-summary { grid-template-columns:1fr; } }
hr { border-color:#edf0f5; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">INTERVIEW STUDIO · AI PRACTICE</div>
  <h1>Show what you know.<br>Build interview confidence.</h1>
  <p>A focused mock interview with spoken questions, adaptive follow-ups, and practical feedback.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ✦ Interview setup")
    st.caption("Personalize your practice session.")
    role = st.selectbox("Target role", [
        "Python Developer", "Django Developer", "Full Stack Developer",
        "Data Scientist", "Machine Learning Engineer", "Software Engineer",
        "DevOps Engineer", "HR / Behavioral Interview"
    ], disabled=st.session_state.started)
    level = st.selectbox("Experience level", ["Fresher", "Junior", "Intermediate", "Senior"], disabled=st.session_state.started)
    st.markdown("**Maximum questions:** 10")
    st.caption("The AI decides when the interview is complete.")
    count = 10
    st.markdown("---")
    st.markdown("**Session format**")
    st.markdown("🎧 Listen to the question\n\n🎙️ Record your answer first\n\n⌨️ Type if you prefer")
    if st.button("Reset session", use_container_width=True):
        reset_interview()
        st.rerun()

if not st.session_state.started:
    col1, col2 = st.columns([1.5, 1], gap="large")
    with col1:
        st.markdown('<div class="panel"><div class="eyebrow">YOUR PRACTICE SESSION</div><h3>One question at a time</h3><p class="small-muted">The interviewer adapts follow-up questions to your answers. After each response, review your score and actionable feedback.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="panel"><h4>Included in your session</h4><p>✓ Role-specific questions</p><p>✓ Voice-first interaction</p><p>✓ Answer scoring and feedback</p><p>✓ Final report to download</p></div>', unsafe_allow_html=True)
    if st.button("Start interview →", type="primary", use_container_width=True):
        try:
            with st.spinner("Preparing your first question..."):
                st.session_state.role = role
                st.session_state.level = level
                st.session_state.max_questions = 10
                st.session_state.question = first_question(role, level)
                st.session_state.started = True
                st.session_state.audio_ready = False
            st.rerun()
        except Exception as e:
            st.error(f"Could not start the interview: {e}")

else:
    total = st.session_state.max_questions
    done = st.session_state.question_number
    st.progress(min(done / total, 1.0))
    st.markdown(
        f"""
        <div class="session-stats">
          <div class="stat-card">
            <div class="stat-label">PROGRESS</div>
            <div class="stat-value">{done}<span> / {total} max</span></div>
            <div class="stat-note">Answers submitted</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">TARGET ROLE</div>
            <div class="stat-value stat-text">{st.session_state.role}</div>
            <div class="stat-note">Interview track</div>
          </div>
          <div class="stat-card">
            <div class="stat-label">EXPERIENCE</div>
            <div class="stat-value stat-text">{st.session_state.level}</div>
            <div class="stat-note">Difficulty level</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.finished:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown(f'<div class="question-label">QUESTION {done + 1} OF {total}</div><div class="question-text">{st.session_state.question}</div>', unsafe_allow_html=True)

        # Generate speech automatically on question arrival, then autoplay.
        if st.session_state.audio_question != st.session_state.question and not st.session_state.audio_bytes:
            try:
                with st.spinner("Preparing your interviewer’s voice..."):
                    st.session_state.audio_bytes = generate_speech(st.session_state.question)
                    st.session_state.audio_question = st.session_state.question
                    st.session_state.audio_ready = False
            except Exception as e:
                st.session_state.audio_error = str(e)
                st.session_state.audio_ready = True  # allow fallback if TTS unavailable

        if st.session_state.audio_bytes:
            audio_b64 = base64.b64encode(st.session_state.audio_bytes).decode("ascii")
            ended = _audio_component(
                audio_b64=audio_b64,
                question_key=f"{done}-{hash(st.session_state.question)}",
                autoplay=True,
                key=f"audio_player_{done}",
                default=False,
            )
            if ended:
                st.session_state.audio_ready = True

        if st.session_state.audio_error:
            st.warning("Audio could not be generated. You can continue using text input.")
        elif not st.session_state.audio_ready:
            st.markdown('<div class="status">🔊 Listen to the full question. Recording and typing unlock when playback finishes.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status ready">✓ Question playback complete. You can answer now.</div>', unsafe_allow_html=True)

        # Priority: recording first; typing is the fallback.
        st.markdown("#### 1 · Record your answer")
        st.caption("Recording becomes available after the interviewer finishes speaking.")
        audio_answer = st.audio_input(
            "Record your answer",
            key=f"record_{done}",
            disabled=not st.session_state.audio_ready,
        )

        st.markdown("#### 2 · Type your answer instead")
        typed_answer = st.text_area(
            "Text answer (optional fallback)",
            key=f"typed_{done}",
            height=120,
            placeholder="If you prefer not to record, type your answer here...",
            disabled=not st.session_state.audio_ready,
        )

        submit = st.button(
            "Submit answer and continue →",
            type="primary",
            use_container_width=True,
            disabled=not st.session_state.audio_ready,
            key=f"submit_{done}",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if submit:
            try:
                if audio_answer is not None:
                    with st.spinner("Transcribing your answer..."):
                        answer = transcribe_audio(audio_answer)
                    st.session_state.last_transcript = answer
                    with st.expander("Detected transcript", expanded=False):
                        st.write(answer)
                else:
                    answer = typed_answer.strip()

                if not answer:
                    st.warning("Record an answer or type a response before continuing.")
                else:
                    with st.spinner("Reviewing your answer and preparing the next question..."):
                        run_answer(answer)
                    st.rerun()
            except Exception as e:
                st.error(f"Could not process your answer: {e}")

    if st.session_state.history:
        last = st.session_state.history[-1]
        ev = last["evaluation"]
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("### Feedback on your last answer")
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(
                f'''<div class="answer-score-card">
                    <div class="answer-score-label">SCORE</div>
                    <div class="answer-score-value">{ev['score']}<span> / 10</span></div>
                </div>''',
                unsafe_allow_html=True,
            )
        c2.markdown(f"**Interviewer feedback**  \n{ev['feedback']}")
        s_col, i_col = st.columns(2)
        with s_col:
            st.markdown("**What went well**")
            for item in ev["strengths"] or ["Keep practising to build consistency."]:
                st.markdown(f"- {item}")
        with i_col:
            st.markdown("**What to improve**")
            for item in ev["improvements"] or ["Practise explaining your reasoning clearly."]:
                st.markdown(f"- {item}")
        if ev["ideal_answer_points"]:
            with st.expander("What a strong answer should include"):
                for item in ev["ideal_answer_points"]:
                    st.markdown(f"- {item}")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.finished:
        scores = [x["evaluation"]["score"] for x in st.session_state.history]
        avg = sum(scores) / len(scores) if scores else 0
        st.balloons()
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("## Interview complete")
        st.markdown(
            f'''
            <div class="score-summary">
                <div class="score-card">
                    <div class="score-label">AVERAGE SCORE</div>
                    <div class="score-number">{avg:.1f}<span>/ 10</span></div>
                    <div class="score-caption">Overall interview performance</div>
                </div>
                <div class="score-card">
                    <div class="score-label">QUESTIONS COMPLETED</div>
                    <div class="score-number">{len(scores)}<span>/ 10</span></div>
                    <div class="score-caption">Questions answered</div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown(st.session_state.final_report)
        report = {
            "role": st.session_state.role,
            "level": st.session_state.level,
            "average_score": round(avg, 2),
            "history": st.session_state.history,
            "final_report": st.session_state.final_report,
        }
        st.download_button(
            "Download interview report (JSON)",
            data=json.dumps(report, indent=2, ensure_ascii=False),
            file_name="interview_report.json",
            mime="application/json",
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        with st.expander("Review all questions and answers"):
            for index, turn in enumerate(st.session_state.history, 1):
                st.markdown(f"**{index}. {turn['question']}**")
                st.write(turn["answer"])
                st.caption(f"Score: {turn['evaluation']['score']}/10 · {turn['evaluation']['feedback']}")
                st.divider()
        if st.button("Start a new interview", type="primary", use_container_width=True):
            reset_interview()
            st.rerun()
