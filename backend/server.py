from flask import Flask, request, jsonify
from flask_cors import CORS
from app import handle_query

app = Flask(__name__)
CORS(app)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_query = data.get("query", "")

    result = handle_query(user_query)

    explanation = result["explanation"]

    return jsonify({
        "answer": result["answer"],
        "reasoning": explanation["reasoning"],
        "confidence": explanation["confidence"],
        "sources": explanation["supporting_sentences"]
    })


if __name__ == "__main__":
    app.run(debug=True)