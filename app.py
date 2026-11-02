import difflib, hashlib, hmac, html, json, os, sqlite3, subprocess, tempfile, unicodedata
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dictation Trainer", page_icon="🎧", layout="centered")

DB = os.environ.get("DB_PATH", "dictation.db")
MIME = {"mp3": "audio/mpeg", "wav": "audio/wav", "m4a": "audio/mp4", "ogg": "audio/ogg", "flac": "audio/flac"}
DEFAULTS = {"lang": "vi", "model": "small", "ig_case": True, "ig_punct": True,
            "ig_accent": False, "loop": False, "goal": 10, "pass_score": 80}
LANGS = ["vi", "en", "ja", "ko", "zh", "fr", "de", "es"]

st.markdown("""<style>
.block-container{padding-top:2rem;max-width:860px}
.hero{background:linear-gradient(135deg,#6366f1,#8b5cf6 60%,#ec4899);color:#fff;
 padding:1.4rem 1.6rem;border-radius:18px;margin-bottom:1rem}
.hero h2{margin:0;color:#fff;font-size:1.6rem} .hero p{margin:.2rem 0 0;opacity:.9}
.result{font-size:1.15rem;line-height:2;padding:.9rem 1.1rem;border-radius:12px;
 background:rgba(99,102,241,.09);margin:.6rem 0}
div[data-testid="stMetric"]{border:1px solid rgba(128,128,128,.25);border-radius:14px;padding:.7rem 1rem}
.stButton>button{border-radius:10px}
</style>""", unsafe_allow_html=True)


# ---------- Database ----------
def q(sql, args=(), one=False):
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    try:
        rows = c.execute(sql, args).fetchall()
        c.commit()
    finally:
        c.close()
    return (rows[0] if rows else None) if one else rows


q("""CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, pw TEXT, salt TEXT,
     name TEXT, settings TEXT, created TEXT DEFAULT (datetime('now','localtime')))""")
q("""CREATE TABLE IF NOT EXISTS lessons(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, title TEXT,
     ext TEXT, audio BLOB, segments TEXT, created TEXT DEFAULT (datetime('now','localtime')))""")
q("""CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, lesson_id INTEGER,
     seg_idx INTEGER, score INTEGER, answer TEXT, ts TEXT DEFAULT (datetime('now','localtime')))""")


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 120_000).hex()


def register(u, pw, name):
    u = u.strip().lower()
    if len(u) < 3 or len(pw) < 6:
        return "Tên đăng nhập ≥ 3 ký tự và mật khẩu ≥ 6 ký tự."
    if q("SELECT 1 FROM users WHERE username=?", (u,), one=True):
        return "Tên đăng nhập đã tồn tại."
    salt = os.urandom(16)
    q("INSERT INTO users(username,pw,salt,name,settings) VALUES(?,?,?,?,?)",
      (u, hash_pw(pw, salt), salt.hex(), name.strip() or u, json.dumps(DEFAULTS)))
    return login(u, pw)


def login(u, pw):
    u = u.strip().lower()
    r = q("SELECT * FROM users WHERE username=?", (u,), one=True)
    if not r or not hmac.compare_digest(r["pw"], hash_pw(pw, bytes.fromhex(r["salt"]))):
        return "Sai tên đăng nhập hoặc mật khẩu."
    ss.user, ss.name = u, r["name"]
    ss.cfg = {**DEFAULTS, **json.loads(r["settings"])}
    ss.page = "home"
    return None


# ---------- Whisper & so sánh ----------
@st.cache_resource(show_spinner="Đang tải model nhận dạng giọng nói...")
def load_model(size):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def decode(path):
    """Giải mã audio bằng ffmpeg đi kèm imageio-ffmpeg -> mảng float32 16 kHz mono (tránh lỗi PyAV)."""
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def transcribe(data, ext, size, lang):
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        segs, _ = load_model(size).transcribe(decode(path), language=lang, vad_filter=True)
        return [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in segs if s.text.strip()]
    finally:
        os.remove(path)


def norm(w, case, punct, accent):
    if case:
        w = w.lower()
    if accent:
        w = unicodedata.normalize("NFD", w.replace("đ", "d").replace("Đ", "D"))
        w = "".join(c for c in w if not unicodedata.combining(c))
    if punct:
        w = "".join(c for c in w if c.isalnum())
    return w


def compare(ref, user, cfg):
    o = (cfg["ig_case"], cfg["ig_punct"], cfg["ig_accent"])
    r = [(w, norm(w, *o)) for w in ref.split()]
    u = [(w, norm(w, *o)) for w in user.split()]
    r, u = [x for x in r if x[1]], [x for x in u if x[1]]
    sm = difflib.SequenceMatcher(None, [k for _, k in r], [k for _, k in u], autojunk=False)
    parts, ok = [], 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ok += i2 - i1
            parts += [f"<span style='color:#2e9e44'>{html.escape(w)}</span>" for w, _ in r[i1:i2]]
        else:
            parts += [f"<span style='color:#d33;text-decoration:line-through'>{html.escape(w)}</span>" for w, _ in u[j1:j2]]
            parts += [f"<b style='color:#e08a00'>{html.escape(w)}</b>" for w, _ in r[i1:i2]]
    return " ".join(parts), round(100 * ok / max(len(r), 1))


# ---------- State & callbacks ----------
ss = st.session_state
for k, v in {"user": None, "page": "home", "draft": None, "idx": 0, "answers": {}, "last": None}.items():
    ss.setdefault(k, v)


def goto(p):
    ss.page = p


def logout():
    for k in list(ss.keys()):
        del ss[k]


def best_scores():
    rows = q("SELECT lesson_id l, seg_idx s, MAX(score) b FROM attempts WHERE username=? GROUP BY l, s", (ss.user,))
    return {(r["l"], r["s"]): r["b"] for r in rows}


def open_lesson(lid):
    r = q("SELECT * FROM lessons WHERE id=? AND username=?", (lid, ss.user), one=True)
    ss.lesson, ss.audio, ss.ext = r["id"], r["audio"], r["ext"]
    ss.segments, ss.title = json.loads(r["segments"]), r["title"]
    b = best_scores()
    ss.idx = next((k for k in range(len(ss.segments)) if b.get((lid, k), 0) < ss.cfg["pass_score"]), 0)
    ss.answers, ss.last, ss.page = {}, None, "practice"


def delete_lesson(lid):
    q("DELETE FROM lessons WHERE id=? AND username=?", (lid, ss.user))
    q("DELETE FROM attempts WHERE lesson_id=? AND username=?", (lid, ss.user))


def step(d):
    ss.idx = max(0, min(len(ss.segments) - 1, ss.idx + d))


# ---------- Đăng nhập ----------
if not ss.user:
    st.markdown("<div class='hero'><h2>🎧 Dictation Trainer</h2>"
                "<p>Nghe – chép – chấm điểm. Đăng nhập để lưu bài học và theo dõi tiến độ của riêng bạn.</p></div>",
                unsafe_allow_html=True)
    t1, t2 = st.tabs(["Đăng nhập", "Tạo tài khoản"])
    with t1, st.form("login"):
        u = st.text_input("Tên đăng nhập")
        p = st.text_input("Mật khẩu", type="password")
        if st.form_submit_button("Đăng nhập", type="primary", use_container_width=True):
            err = login(u, p)
            st.error(err) if err else st.rerun()
    with t2, st.form("reg"):
        n = st.text_input("Tên hiển thị")
        u = st.text_input("Tên đăng nhập mới")
        p = st.text_input("Mật khẩu (≥ 6 ký tự)", type="password")
        if st.form_submit_button("Tạo tài khoản", type="primary", use_container_width=True):
            err = register(u, p, n)
            st.error(err) if err else st.rerun()
    st.stop()

cfg = ss.cfg

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(f"### 👋 {ss.name}")
    for key, label in [("home", "📚 Bài học"), ("new", "➕ Tạo bài mới"),
                       ("stats", "📊 Thống kê"), ("settings", "⚙️ Cài đặt")]:
        st.button(label, key=f"nav_{key}", on_click=goto, args=(key,), use_container_width=True,
                  type="primary" if ss.page == key else "secondary")
    st.divider()
    st.button("Đăng xuất", on_click=logout, use_container_width=True)

# ---------- Trang: Bài học ----------
if ss.page == "home":
    today = q("SELECT COUNT(*) n FROM attempts WHERE username=? AND date(ts)=date('now','localtime')",
              (ss.user,), one=True)["n"]
    st.markdown(f"<div class='hero'><h2>Chào {html.escape(ss.name)} 👋</h2>"
                f"<p>Hôm nay bạn đã luyện {today}/{cfg['goal']} câu.</p></div>", unsafe_allow_html=True)
    st.progress(min(today / cfg["goal"], 1.0), text="Mục tiêu hằng ngày")
    lessons = q("SELECT id,title,created,segments FROM lessons WHERE username=? ORDER BY id DESC", (ss.user,))
    if not lessons:
        st.info("Bạn chưa có bài nào. Vào **➕ Tạo bài mới** để tải file nghe lên.")
    b = best_scores()
    for L in lessons:
        n = len(json.loads(L["segments"]))
        done = sum(1 for k in range(n) if b.get((L["id"], k), 0) >= cfg["pass_score"])
        with st.container(border=True):
            a, c1, c2 = st.columns([6, 2, 1])
            a.markdown(f"**{L['title']}**")
            a.caption(f"{n} câu · tạo {L['created'][:10]}")
            a.progress(done / max(n, 1), text=f"{done}/{n} câu đạt ≥ {cfg['pass_score']}%")
            c1.button("▶️ Luyện", key=f"go{L['id']}", on_click=open_lesson, args=(L["id"],),
                      type="primary", use_container_width=True)
            c2.button("🗑️", key=f"del{L['id']}", on_click=delete_lesson, args=(L["id"],))

# ---------- Trang: Tạo bài ----------
elif ss.page == "new":
    st.header("➕ Tạo bài mới")
    title = st.text_input("Tên bài")
    up = st.file_uploader("File nghe", type=list(MIME))
    if up and st.button("✂️ Tự động tách câu", type="primary"):
        with st.spinner("Đang nhận dạng và tách câu, file dài có thể mất vài phút..."):
            ext = up.name.rsplit(".", 1)[-1].lower()
            ss.draft = {"audio": up.getvalue(), "ext": ext,
                        "segments": transcribe(up.getvalue(), ext, cfg["model"], cfg["lang"]),
                        "name": up.name}
    d = ss.draft
    if d:
        st.caption("Sửa lại văn bản chuẩn nếu Whisper nghe sai; có thể chỉnh mốc thời gian, thêm/xóa dòng.")
        df = st.data_editor(pd.DataFrame(d["segments"]), num_rows="dynamic",
                            use_container_width=True, hide_index=True)
        if st.button("💾 Lưu bài học", type="primary"):
            df = df.dropna(subset=["start", "end", "text"]).sort_values("start")
            q("INSERT INTO lessons(username,title,ext,audio,segments) VALUES(?,?,?,?,?)",
              (ss.user, title.strip() or d["name"], d["ext"], d["audio"], json.dumps(df.to_dict("records"))))
            ss.draft, ss.page = None, "home"
            st.rerun()

# ---------- Trang: Luyện ----------
elif ss.page == "practice":
    segs, i = ss.segments, ss.idx
    seg = segs[i]
    st.button("← Danh sách bài", on_click=goto, args=("home",))
    st.subheader(ss.title)
    b = best_scores()
    st.progress((i + 1) / len(segs), text=f"Câu {i + 1}/{len(segs)}"
                + (f" · điểm cao nhất {b[(ss.lesson, i)]}%" if (ss.lesson, i) in b else ""))
    st.audio(ss.audio, format=MIME.get(ss.ext, "audio/mpeg"),
             start_time=max(0.0, float(seg["start"]) - 0.15), end_time=float(seg["end"]) + 0.25,
             loop=cfg["loop"])
    ans = st.text_area("Nghe và chép lại:", value=ss.answers.get(i, ""), key=f"ans_{ss.lesson}_{i}", height=90)
    ss.answers[i] = ans
    c1, c2, c3, c4 = st.columns(4)
    c1.button("⬅️ Trước", on_click=step, args=(-1,), disabled=i == 0, use_container_width=True)
    check = c2.button("✅ Kiểm tra", type="primary", use_container_width=True)
    hint = c3.toggle("💡 Gợi ý")
    c4.button("Sau ➡️", on_click=step, args=(1,), disabled=i == len(segs) - 1, use_container_width=True)
    if hint:
        st.caption(" ".join(w[0] + "_" * (len(w) - 1) for w in seg["text"].split()))
    if check:
        diff, score = compare(seg["text"], ans, cfg)
        q("INSERT INTO attempts(username,lesson_id,seg_idx,score,answer) VALUES(?,?,?,?,?)",
          (ss.user, ss.lesson, i, score, ans))
        ss.last = (i, diff, score)
        if score == 100:
            st.toast("Chính xác tuyệt đối! 🎉")
    if ss.last and ss.last[0] == i:
        _, diff, score = ss.last
        st.markdown(f"<div class='result'>{diff}</div>", unsafe_allow_html=True)
        st.caption("🟢 đúng · đỏ gạch ngang: chép sai/thừa · cam đậm: từ đúng bạn bỏ sót hoặc chép sai")
        (st.success if score >= cfg["pass_score"] else st.warning if score >= 50 else st.error)(
            f"Độ chính xác: {score}%")

# ---------- Trang: Thống kê ----------
elif ss.page == "stats":
    st.header("📊 Thống kê")
    rows = q("SELECT date(ts) d, COUNT(*) n, AVG(score) a FROM attempts WHERE username=? GROUP BY d ORDER BY d",
             (ss.user,))
    if not rows:
        st.info("Chưa có dữ liệu. Hãy luyện vài câu nhé!")
    else:
        days = {r["d"] for r in rows}
        cur, streak = date.today(), 0
        if cur.isoformat() not in days:
            cur -= timedelta(days=1)
        while cur.isoformat() in days:
            streak, cur = streak + 1, cur - timedelta(days=1)
        tot = sum(r["n"] for r in rows)
        avg = sum(r["a"] * r["n"] for r in rows) / tot
        m = st.columns(3)
        m[0].metric("🔥 Chuỗi ngày", streak)
        m[1].metric("Lượt chấm", tot)
        m[2].metric("Điểm TB", f"{avg:.0f}%")
        df = pd.DataFrame(rows[-14:], columns=["Ngày", "Số câu", "Điểm TB"]).set_index("Ngày")
        st.markdown("**Số câu luyện mỗi ngày**")
        st.bar_chart(df["Số câu"])
        st.markdown("**Điểm trung bình mỗi ngày**")
        st.line_chart(df["Điểm TB"])
        weak = q("""SELECT l.title t, a.seg_idx s, MIN(a.score) m FROM attempts a JOIN lessons l ON l.id=a.lesson_id
                    WHERE a.username=? GROUP BY a.lesson_id, a.seg_idx HAVING MAX(a.score) < ? ORDER BY m LIMIT 5""",
                 (ss.user, cfg["pass_score"]))
        if weak:
            st.markdown("**Câu cần ôn lại**")
            for r in weak:
                st.write(f"• {r['t']} – câu {r['s'] + 1}")

# ---------- Trang: Cài đặt ----------
elif ss.page == "settings":
    st.header("⚙️ Cài đặt cá nhân")
    with st.form("settings"):
        name = st.text_input("Tên hiển thị", ss.name)
        c1, c2 = st.columns(2)
        lang = c1.selectbox("Ngôn ngữ audio mặc định", LANGS, index=LANGS.index(cfg["lang"]))
        models = ["tiny", "base", "small", "medium"]
        model = c2.selectbox("Model Whisper", models, index=models.index(cfg["model"]))
        goal = c1.number_input("Mục tiêu mỗi ngày (câu)", 1, 200, cfg["goal"])
        ps = c2.slider("Điểm đạt (%)", 50, 100, cfg["pass_score"])
        ig_case = st.checkbox("Bỏ qua hoa/thường", cfg["ig_case"])
        ig_punct = st.checkbox("Bỏ qua dấu câu", cfg["ig_punct"])
        ig_accent = st.checkbox("Bỏ qua dấu thanh (á, à, ...)", cfg["ig_accent"])
        loop = st.checkbox("Lặp lại câu đang nghe", cfg["loop"])
        if st.form_submit_button("💾 Lưu", type="primary"):
            ss.cfg = dict(lang=lang, model=model, goal=int(goal), pass_score=ps, ig_case=ig_case,
                          ig_punct=ig_punct, ig_accent=ig_accent, loop=loop)
            ss.name = name.strip() or ss.user
            q("UPDATE users SET name=?, settings=? WHERE username=?", (ss.name, json.dumps(ss.cfg), ss.user))
            st.success("Đã lưu cài đặt.")
