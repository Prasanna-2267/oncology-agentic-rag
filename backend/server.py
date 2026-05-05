from flask import Flask, request, jsonify
from flask_cors import CORS
from app import handle_query

app = Flask(__name__)
CORS(app)


# -------------------------------
# 🔹 MAIN API
# -------------------------------
@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.get_json()

        # 🔥 Validate input
        if not data or "query" not in data:
            return jsonify({"error": "Missing query"}), 400

        user_query = data.get("query", "").strip()

        if not user_query:
            return jsonify({"error": "Empty query"}), 400

        # 🔹 Run pipeline
        result = handle_query(user_query)

        explanation = result.get("explanation", {})

        return jsonify({
            "answer": result.get("answer", ""),
            "confidence": result.get("confidence", 0.75),

            # 🔥 UI-friendly
            "reasoning": explanation.get("reasoning", ""),
            "supporting_sentences": explanation.get("supporting_sentences", []),

            # 🔥 REAL SOURCES (IMPORTANT)
            "sources": result.get("sources", []),            # doc_ids
            "source_texts": result.get("source_texts", [])   # actual text
        })

    except Exception as e:
        print("❌ SERVER ERROR:", e)

        return jsonify({
            "answer": "Something went wrong. Please try again.",
            "confidence": 0.5,
            "reasoning": "System error occurred.",
            "supporting_sentences": [],
            "sources": [],
            "source_texts": []
        }), 500


# -------------------------------
# 🔹 HEALTH CHECK
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# -------------------------------
# 🔹 RUN SERVER
# -------------------------------
if __name__ == "__main__":
    print("🚀 Server running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)