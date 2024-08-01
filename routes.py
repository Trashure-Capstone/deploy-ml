from flask import Flask, render_template, request, jsonify, current_app
from prediction.inference import class_names, plastic_dict, predict_label
from prediction.train_stream import generate_dataset_dir
from database.models import (
    get_history,
    save_history,
    update_history,
    add_feedback,
    get_count_feedback,
    get_batch_data,
)
import shutil
from io import BytesIO
import cv2
import numpy as np
import os
import time, datetime

app = Flask(__name__, static_folder="static", static_url_path="/static")

app.config.from_pyfile("config.py")


def duplicate_image(filename):
    # copy image from static/data/prediction to static/data/feedback
    src = os.path.join(app.config["PREDICTION_DIR"], filename)
    dst = os.path.join(app.config["FEEDBACK_DIR"], filename)
    shutil.copyfile(src, dst)
    return dst


@app.route("/", methods=["GET"])
def home():
    page = request.args.get("page", 1, type=int)
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    history = get_history()[start:end]

    total_history = len(get_history())
    total_pages = (total_history + per_page - 1) // per_page

    # Tambahkan per_page ke render_template
    return render_template(
        "index.html",
        history=history,
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        app=app,
    )


@app.route("/monitoring")
def monitoring():
    page = request.args.get("page", 1, type=int)
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    batch_data = get_batch_data()[start:end]

    feedback_data = get_count_feedback()

    total_batch_data = len(get_batch_data())
    total_pages = (total_batch_data + per_page - 1) // per_page

    return render_template(
        "monitoring.html",
        batch_data=batch_data,
        feedback_data=feedback_data,
        page=page,
        total_pages=total_pages,
        per_page=per_page,
        app=app,
    )


@app.route("/update_identification", methods=["POST"])
def update():
    if request.method == "POST":
        data = request.get_json()
        data["update_at"] = datetime.datetime.now()
        data["is_correct"] = False
        filename = data["image_path"].split("/")[-1]
        update_history(data)

        # recent_batch = get_recent_batch() # get_recent_batch function where if training is done, it will return the recent batch id
        data["batch"] = 1
        image_path = duplicate_image(filename)
        data["image_path"] = image_path
        add_feedback(data)

        return jsonify({"success": True}), 200
    return jsonify({"error": True, "message": "Method not allowed"}), 405


# @app.route("/update_review_status/<int:history_id>", methods=["POST"])
# def update_review_status(history_id):
#     history = History.query.get(history_id)
#     if history:
#         history.is_reviewed = True
#         db.session.commit()
#         return jsonify({"success": True})
#     else:
#         return jsonify({"success": False, "error": "History not found"}), 404

# seperate the route and the function, the function is in controller.py


@app.route("/scan", methods=["POST"])
def predict():
    if request.method == "POST":
        the_file = request.files["file"]

        file_stream = BytesIO(the_file.read())
        image = cv2.imdecode(
            np.frombuffer(file_stream.read(), np.uint8), cv2.IMREAD_COLOR
        )

        # Ensure the directory exists
        save_dir = r"static/data/prediction/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Generate a unique filename for the image
        save_path = os.path.join(save_dir, the_file.filename)

        # Save image to the specified path
        cv2.imwrite(save_path, image)
        predicted_image = predict_label(image)

        max_prob = predicted_image["confidence"]
        identification = predicted_image["identification"]
        processing_times = predicted_image["processingTime"]

        path = save_dir.split("static/")[-1] + the_file.filename
        save_history(
            {
                "path": path,
                "identification": identification,
                "confidence": float(max_prob),
                "processing_times": processing_times,
                "is_correct": True,
            }
        )

        # full_info = plastic_dict[max_prob_class] # Mengambil informasi lengkap dari dictionary
        output = {
            "identification": identification,
            "confidence": float(max_prob),
            "processingTime": float(processing_times),
        }
        return jsonify({"result": output}), 200

    return jsonify({"error": True, "message": "Method not allowed"}), 405


@app.route("/test", methods=["GET"])
def test():
    data_split_ratio = {
        "train": 0.8,
        "validation": 0.1,
        "test": 0.1,
    }
    feeds = generate_dataset_dir(0, data_split_ratio)
    return jsonify({"feeds": feeds})


if __name__ == "__main__":
    app.run(debug=True)
