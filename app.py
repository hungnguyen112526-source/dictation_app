import difflib
import html
import os
import tempfile
import unicodedata

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dictation Trainer", page_icon="🎧", layout="centered")

MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4",
        "ogg": "audio/ogg", "flac": "audio/flac"}


# ---------- Xử lý âm thanh / văn bản ----------
@st.cache_resource(show_spinner="Đang tải model nhận dạng giọng nói...")
def load_model(size: str):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(data: bytes, ext: str, size: str, lang: str):
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        segs, _ = load_model(size).transcribe(path, language=lang or None, vad_filter=True)
        return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segs if s.text.strip()]
    finally:
        os.remove(path)


def norm(word, ignore_case, ignore_punct, ignore_accent):
    if ignore_case:
        word = word.lower()
    if ignore_accent:
        word = unicodedata.normalize("NFD", word.replace("đ", "d").replace("Đ", "D"))
        word = "".join(c for c in word if not unicodedata.combining(c))
    if ignore_punct:
        word = "".join(c for c in word if c.isalnum())
    return word


def words(text, *opts):
    out = []
    for w in text.split():
        k = norm(w, *opts)
        if k:
            out.append((w, k))
    return out


def compare(ref, user, opts):
    r, u = words(ref, *opts), words(user, *opts)
    sm = difflib.SequenceMatcher(None, [k for _, k in r], [k for _, k in u], autojunk=False)
    parts, correct = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            correct += i2 - i1
            parts += [f"<span style='color:#2e9e44'>{html.escape(w)}</span>" for w, _ in r[i1:i2]]
        else:
            parts += [f"<span style='color:#d33;text-decoration:line-through'>{html.escape(w)}</span>"
                      for w, _ in u[j1:j2]]
            parts += [f"<b style='color:#e08a00'>{html.escape(w)}</b>" for w, _ in r[i1:i2]]
    score = round(100 * correct / max(len(r), 1))
    return " ".join(parts), score


# ---------- State ----------
ss = st.session_state
for k, v in {"audio": None, "ext": "mp3", "segments": [], "practice": False,
             "idx": 0, "answers": {}, "scores": {}}.items():
    ss.setdefault(k, v)


def go(delta):
    ss.idx = max(0, min(len(ss.segments) - 1, ss.idx + delta))


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Cài đặt")
    model_size = st.selectbox("Model Whisper", ["tiny", "base", "small", "medium"], index=2,
                              help="Tiếng Việt nên dùng small/medium trở lên.")
    lang = st.selectbox("Ngôn ngữ audio", ["vi", "en", "ja", "ko", "zh", "fr", "de", "es"])
    st.subheader("Cách chấm")
    ig_case = st.checkbox("Bỏ qua hoa/thường", True)
    ig_punct = st.checkbox("Bỏ qua dấu câu", True)
    ig_accent = st.checkbox("Bỏ qua dấu thanh (á, à, ...)", False)
    loop = st.checkbox("Lặp lại câu đang nghe", False)
    if ss.practice and st.button("↩️ Về màn hình chuẩn bị"):
        ss.practice = False
        st.rerun()

st.title("🎧 Dictation Trainer")

# ---------- Bước 1: chuẩn bị ----------
if not ss.practice:
    up = st.file_uploader("Tải file nghe", type=list(MIME))
    if up is not None and (ss.audio is None or up.getvalue() != ss.audio):
        ss.audio, ss.ext = up.getvalue(), up.name.rsplit(".", 1)[-1].lower()
        ss.segments, ss.answers, ss.scores, ss.idx = [], {}, {}, 0

    if ss.audio:
        st.audio(ss.audio, format=MIME.get(ss.ext, "audio/mpeg"))
        if st.button("✂️ Tự động tách câu (Whisper)", type="primary"):
            with st.spinner("Đang nhận dạng và tách câu, file dài có thể mất vài phút..."):
                ss.segments = transcribe(ss.audio, ss.ext, model_size, lang)

    if ss.segments:
        st.subheader("Kiểm tra & chỉnh sửa")
        st.caption("Sửa lại văn bản chuẩn nếu Whisper nghe sai. Có thể chỉnh mốc thời gian, "
                   "thêm hoặc xóa dòng.")
        df = st.data_editor(pd.DataFrame(ss.segments), num_rows="dynamic",
                            use_container_width=True, hide_index=True)
        if st.button("▶️ Bắt đầu luyện", type="primary"):
            df = df.dropna(subset=["start", "end", "text"])
            ss.segments = df.sort_values("start").to_dict("records")
            ss.practice, ss.idx, ss.answers, ss.scores = True, 0, {}, {}
            st.rerun()
    elif not ss.audio:
        st.info("Tải một file âm thanh lên để bắt đầu.")
    st.stop()

# ---------- Bước 2: luyện ----------
segs, i = ss.segments, ss.idx
seg = segs[i]
st.progress((i + 1) / len(segs), text=f"Câu {i + 1}/{len(segs)}")

st.audio(ss.audio, format=MIME.get(ss.ext, "audio/mpeg"),
         start_time=max(0.0, float(seg["start"]) - 0.15),
         end_time=float(seg["end"]) + 0.25, loop=loop, key=f"audio_{i}")

ans = st.text_area("Nghe và chép lại:", value=ss.answers.get(i, ""), key=f"ans_{i}", height=90)
ss.answers[i] = ans

c1, c2, c3, c4 = st.columns(4)
c1.button("⬅️ Câu trước", on_click=go, args=(-1,), disabled=i == 0, use_container_width=True)
check = c2.button("✅ Kiểm tra", type="primary", use_container_width=True)
hint = c3.toggle("💡 Gợi ý")
c4.button("Câu sau ➡️", on_click=go, args=(1,), disabled=i == len(segs) - 1,
          use_container_width=True)

if hint:
    st.caption(" ".join(w[0] + "_" * (len(w) - 1) for w in seg["text"].split()))

if check:
    diff, score = compare(seg["text"], ans, (ig_case, ig_punct, ig_accent))
    ss.scores[i] = score
    st.markdown(diff, unsafe_allow_html=True)
    st.caption("🟢 đúng · 🔴 gạch ngang: bạn chép sai/thừa · 🟠 đậm: từ đúng bạn bỏ sót hoặc chép sai")
    (st.success if score >= 90 else st.warning if score >= 60 else st.error)(f"Độ chính xác: {score}%")

# ---------- Tổng kết ----------
if ss.scores:
    with st.expander(f"📊 Tiến độ ({len(ss.scores)}/{len(segs)} câu đã chấm)"):
        st.metric("Điểm trung bình", f"{sum(ss.scores.values()) / len(ss.scores):.0f}%")
        st.dataframe(pd.DataFrame({"Câu": [k + 1 for k in ss.scores],
                                   "Điểm (%)": list(ss.scores.values())}),
                     hide_index=True, use_container_width=True)
