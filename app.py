"""
app.py - Générateur de CV LaTeX personnalisé (Streamlit) avec OpenAI >=1.0

Usage:
- Déposer ce fichier dans ton repo avec `template.tex`, `cv_master.json` et ta photo (ex: mouad.png).
- Déployer sur Streamlit Cloud.
- Configurer OPENAI_API_KEY dans les Secrets (Settings -> Secrets).

Dépendances (requirements.txt):
streamlit
openai>=1.0.0
rapidfuzz
"""

import streamlit as st
import json
import re
import io
import os
from rapidfuzz import fuzz
import openai

# ---------------------------------------
# Config
# ---------------------------------------
DEFAULT_MODEL = "gpt-4o-mini"  # tu peux changer
TEMPLATE_PATH = "template.tex"
DEFAULT_CV_PATH = "cv_master.json"

st.set_page_config(page_title="Générateur de CV - LaTeX", layout="wide")

# ---------------------------------------
# Helper - OpenAI key
# ---------------------------------------
def init_openai():
    if "OPENAI_API_KEY" in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
    elif os.getenv("OPENAI_API_KEY"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
    else:
        openai.api_key = None

init_openai()

# ---------------------------------------
# Utilities
# ---------------------------------------
def tex_escape(s: str) -> str:
    if not isinstance(s, str):
        return s
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&","\\&").replace("%","\\%").replace("$","\\$")
             .replace("#","\\#").replace("_","\\_").replace("{","\\{")
             .replace("}","\\}").replace("~","\\textasciitilde{}")
             .replace("^","\\textasciicircum{}"))

def load_json_file(uploaded_file):
    try:
        if uploaded_file is None:
            return None
        return json.load(uploaded_file)
    except Exception as e:
        st.error(f"Impossible de lire le JSON : {e}")
        return None

def safe_extract_json(text):
    """Extrait le premier objet/array JSON trouvé dans un texte."""
    m = re.search(r'(\{.*\}|\[.*\])', text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return None

# ---------------------------------------
# OpenAI wrappers (nouvelle API >=1.0)
# ---------------------------------------
def openai_chat_completion(system, user_prompt, model=DEFAULT_MODEL, temperature=0.0, max_tokens=800):
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY introuvable. Ajoute-la dans Streamlit secrets ou en variable d'environnement.")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_prompt})

    resp = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

def extract_keywords_from_job(job_text):
    system = "Tu es un assistant qui extrait des mots-clés techniques et soft-skills d'une offre d'emploi/stage. Répond uniquement avec une liste JSON (array) de mots-clés courts."
    user = f"Offre:\n{job_text}\n\nRetourne une liste JSON des mots-clés (langages, outils, domaines, soft skills)."
    raw = openai_chat_completion(system, user, temperature=0.0)
    parsed = safe_extract_json(raw)
    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if x]
    items = re.split(r'[\n,;]+', re.sub(r'[^0-9A-Za-zÀ-ÖØ-öø-ÿ_\-.,; ]', ' ', raw))
    return [i.strip() for i in items if i.strip()][:30]

def rewrite_items_openai(items, job_keywords, section_name, model=DEFAULT_MODEL):
    system = "Tu es un rédacteur professionnel de CV. Réécris les titres et bullets pour qu'ils soient adaptés à l'offre. Tu peux reformuler de façon concise, mais reste fidèle à la réalité et ne crée pas d'expériences nouvelles."
    user = (
        f"Contexte - mots-clés de l'offre: {job_keywords}\n"
        f"Section: {section_name}\n"
        f"Éléments à reformuler (JSON):\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        "Retourne STRICTEMENT un JSON (liste) de la même longueur contenant des objets {title, company, tasks}."
    )
    raw = openai_chat_completion(system, user, temperature=0.25)
    parsed = safe_extract_json(raw)
    if isinstance(parsed, list):
        norm = []
        for i, obj in enumerate(parsed):
            title = obj.get("title") or items[i].get("title") if isinstance(obj, dict) else items[i].get("title")
            company = obj.get("company") or items[i].get("company","")
            tasks = obj.get("tasks") or items[i].get("tasks",[])
            if isinstance(tasks, str):
                tasks = [t.strip() for t in re.split(r'[\n•\-]+', tasks) if t.strip()]
            norm.append({"title": title, "company": company, "tasks": tasks})
        return norm
    fallback = []
    for it in items:
        fallback.append({
            "title": it.get("title",""),
            "company": it.get("company",""),
            "tasks": it.get("tasks", [])
        })
    return fallback

def generate_profile_text(cv_data, job_text, job_keywords):
    system = "Tu es un rédacteur professionnel. Rédige un paragraphe de profil CV (3 à 5 phrases) en français, concis et orienté résultat, adapté à l'offre."
    user = (
        f"CV résumé (sans données sensibles): {json.dumps({'stages': cv_data.get('stages',[]),'competences': cv_data.get('competences',[])}, ensure_ascii=False)[:1500]}\n\n"
        f"Offre:\n{job_text[:2500]}\n\n"
        f"Mots-clés: {job_keywords}\n\n"
        "Écris un paragraphe adapté, professionnel, en français. Ne change pas la réalité et n'ajoute pas de faits nouveaux."
    )
    raw = openai_chat_completion(system, user, temperature=0.45, max_tokens=200)
    return raw.strip()

# ---------------------------------------
# Relevance & formatting
# ---------------------------------------
def relevance_score(item, keywords):
    text_pool = " ".join([
        str(item.get("title","")),
        str(item.get("description","") or ""),
        " ".join(item.get("tasks",[]) or []),
        " ".join(item.get("keywords",[]) or [])
    ]).lower()
    score = 0
    for k in keywords:
        score += fuzz.partial_ratio(k.lower(), text_pool)
    return score

def select_top(items, keywords, n=3):
    if not items:
        return []
    scored = [(relevance_score(it, keywords), it) for it in items]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:n]]

def format_experience_tex(exp_list):
    lines = []
    for e in exp_list:
        title = tex_escape(e.get("title",""))
        company = tex_escape(e.get("company",""))
        start = e.get("start","")
        end = e.get("end","")
        period = f"{start} -- {end}" if start or end else ""
        lines.append(f"\\cventry{{{tex_escape(period)}}}{{{title}}}{{{company}}}{{}}{{}}{{\\begin{{itemize}}")
        tasks = e.get("tasks",[]) or []
        for t in tasks:
            lines.append(f"\\item {tex_escape(t)}")
        lines.append("\\end{itemize}}}")
    return "\n".join(lines)

def format_projets_tex(projets):
    lines = []
    for p in projets:
        title = tex_escape(p.get("title",""))
        desc = tex_escape(p.get("description",""))
        lines.append(f"\\cventry{{}}{{{title}}}{{Projet académique}}{{}}{{}}{{\\begin{{itemize}}")
        for line in re.split(r'[\n\r]+', desc):
            if line.strip():
                lines.append(f"\\item {line.strip()}")
        lines.append("\\end{itemize}}}")
    return "\n".join(lines)

def format_competences_tex(skills):
    tags = []
    for s in skills:
        name = s.get("name") if isinstance(s, dict) else s
        if name:
            tags.append(f"\\cvtag{{{tex_escape(str(name))}}}")
    return " ".join(tags)

# ---------------------------------------
# Streamlit UI
# ---------------------------------------
st.title("🧠 Générateur de CV personnalisé (LaTeX)")

st.sidebar.header("Paramètres")
model = st.sidebar.text_input("Modèle OpenAI", DEFAULT_MODEL)
n_exps = st.sidebar.number_input("Nombre d'expériences à afficher", min_value=1, max_value=6, value=3, step=1)
n_projs = st.sidebar.number_input("Nombre de projets à afficher", min_value=0, max_value=6, value=2, step=1)

st.markdown(
    "Colle l'offre de stage (ou upload) puis upload ton `cv_master.json`. "
    "L'application génèrera un fichier `CV_Personnalise.tex` que tu pourras télécharger. "
    "Sur Streamlit Cloud on ne compile pas le PDF : télécharge le .tex et compile localement avec `pdflatex`."
)

job_text = st.text_area("✉️ Offre de stage (texte complet)", height=300, placeholder="Colle ici le texte complet de l'offre...")
uploaded_cv = st.file_uploader("📄 Upload ton `cv_master.json`", type=["json"])

cv_data = None
if uploaded_cv is not None:
    cv_data = load_json_file(uploaded_cv)
else:
    if os.path.exists(DEFAULT_CV_PATH):
        with open(DEFAULT_CV_PATH, "r", encoding="utf-8") as f:
            cv_data = json.load(f)
        st.info(f"Utilisation du `{DEFAULT_CV_PATH}` présent dans le repo.")
    else:
        st.info("Aucun `cv_master.json` uploadé et aucun fichier par défaut trouvé dans le repo.")

template_override = st.file_uploader("Option: uploader un template.tex personnalisé (optionnel)", type=["tex"])
if template_override is not None:
    template_text = template_override.read().decode("utf-8")
else:
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template_text = f.read()
    else:
        template_text = None
        st.error("template.tex introuvable dans le repo.")

generate = st.button("🚀 Générer le CV LaTeX")

if generate:
    if not job_text or not cv_data or not template_text:
        st.error("Merci de fournir l'offre, le cv_master.json et template.tex.")
    else:
        DEFAULT_MODEL = model
        init_openai()
        if not openai.api_key:
            st.error("OPENAI_API_KEY introuvable. Ajoute la clé dans Streamlit Secrets (Settings -> Secrets).")
        else:
            with st.spinner("Analyse de l'offre et génération en cours (API OpenAI)..."):
                try:
                    keywords = extract_keywords_from_job(job_text)
                except Exception as e:
                    st.error(f"Erreur extraction mots-clés: {e}")
                    keywords = []

                st.markdown("**Mots-clés détectés :**")
                st.write(keywords)

                stages = cv_data.get("stages", []) or cv_data.get("stages", [])
                projets = cv_data.get("projets", []) or cv_data.get("projects", []) or []
                selected_stages = select_top(stages, keywords, n=n_exps)
                selected_projs = select_top(projets, keywords, n=n_projs)

                st.markdown("**Expériences sélectionnées (avant réécriture)**")
                for s in selected_stages:
                    st.write(f"- {s.get('title','')} — {s.get('company','')}")
                st.markdown("**Projets sélectionnés (avant réécriture)**")
                for p in selected_projs:
                    st.write(f"- {p.get('title','')}")

                try:
                    adapted_stages = rewrite_items_openai(selected_stages, keywords, "Expériences")
                    adapted_projs = rewrite_items_openai(selected_projs, keywords, "Projets")
                except Exception as e:
                    st.error(f"Erreur réécriture OpenAI : {e}")
                    adapted_stages = selected_stages
                    adapted_projs = selected_projs

                st.markdown("**Expériences après réécriture (aperçu)**")
                for s in adapted_stages:
                    st.write(f"**{s.get('title','')}** — {s.get('company','')}")
                    for t in s.get("tasks", [])[:5]:
                        st.write(f"• {t}")

                try:
                    profile_text = generate_profile_text(cv_data, job_text, keywords)
                except Exception as e:
                    st.error(f"Erreur génération profil: {e}")
                    profile_text = cv_data.get("person", {}).get("summary", "") or ""

                st.markdown("**Profil généré :**")
                st.info(profile_text)

                tex_filled = template_text
                tex_filled = tex_filled.replace("{{{PROFIL}}}", tex_escape(profile_text))
                tex_filled = tex_filled.replace("{{EXPERIENCES}}", format_experience_tex(adapted_stages))
                tex_filled = tex_filled.replace("{{PROJETS}}", format_projets_tex(adapted_projs))
                tex_filled = tex_filled.replace("{{COMPETENCES}}", format_competences_tex(cv_data.get("competences", [])))

                st.subheader("Prévisualisation du LaTeX généré")
                st.code(tex_filled[:4000] + ("\n\n... (troncated)" if len(tex_filled) > 4000 else ""), language="latex")

                tex_bytes = tex_filled.encode("utf-8")
                tex_buf = io.BytesIO(tex_bytes)
                st.download_button(
                    label="📄 Télécharger le fichier LaTeX (.tex)",
                    data=tex_buf,
                    file_name="CV_Personnalise.tex",
                    mime="application/x-tex"
                )

                review_payload = {
                    "keywords": keywords,
                    "adapted_stages": adapted_stages,
                    "adapted_projets": adapted_projs,
                    "profile": profile_text
                }
                review_bytes = io.BytesIO(json.dumps(review_payload, ensure_ascii=False, indent=2).encode("utf-8"))
                st.download_button(
                    label="🔍 Télécharger le résumé JSON (revue avant envoi)",
                    data=review_bytes,
                    file_name="cv_adapted_review.json",
                    mime="application/json"
                )

                st.success("Fichier LaTeX généré — télécharge le fichier et compile localement pour obtenir le PDF (pdflatex).")
