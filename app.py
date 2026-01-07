import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import io
import base64

app = Flask(__name__)

# -----------------------------
# Load Malayalam Word2Vec model
# -----------------------------
MODEL_PATH = os.path.join("models", "ml_small.kv")
model = KeyedVectors.load(MODEL_PATH)

print("✅ Malayalam Word2Vec model loaded")
print("📘 Vocabulary size:", len(model))

# -----------------------------
# Load Malayalam-compatible font
# -----------------------------
FONT_PATH = os.path.join("Font", "kartika.ttf")
malayalam_font = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    img_data = None

    if request.method == "POST":
        word = request.form.get("word", "").strip()

        try:
            topn = int(request.form.get("topn", 5))
        except ValueError:
            topn = 5

        if not word:
            result = "Please enter a word."
        elif word not in model:
            result = "Word not found in vocabulary."
        else:
            similar_words = model.most_similar(word, topn=topn)

            words = [word] + [w for w, _ in similar_words]
            vectors = [model[w] for w in words]

            # PCA visualization
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(vectors)

            fig, ax = plt.subplots(figsize=(8, 6))

            for i, w in enumerate(words):
                color = "red" if i == 0 else "blue"
                ax.scatter(reduced[i, 0], reduced[i, 1], color=color)
                ax.annotate(
                    w,
                    (reduced[i, 0] + 0.01, reduced[i, 1] + 0.01),
                    fontproperties=malayalam_font,
                    fontsize=12,
                    color=color
                )

            ax.set_title(
                f"Top {topn} Similar Words to '{word}'",
                fontproperties=malayalam_font
            )
            ax.grid(True)

            # Save plot to memory
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close()

            result = [(w, f"{s:.4f}") for w, s in similar_words]

    return render_template(
        "malayalam_index.html",
        result=result,
        image=img_data
    )

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5003)
