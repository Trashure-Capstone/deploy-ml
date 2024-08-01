from database.db_conn import get_db_connection


def get_history():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM history order by created_at desc")
    history = cursor.fetchall()
    cursor.close()
    conn.close()
    return history


def save_history(data):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "INSERT INTO history (path, identification, confidence, processing_times, is_correct, created_at) VALUES (%s, %s, %s, %s, %s, NOW())",
        (
            data["path"],
            data["identification"],
            data["confidence"],
            data["processing_times"],
            data["is_correct"],
        ),
    )
    conn.commit()
    cursor.close()
    conn.close()


def update_history(data):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "UPDATE history SET is_correct=%s, new_identification=%s, update_at=%s WHERE id=%s",
        (data["is_correct"], data["new_identification"], data["update_at"], data["id"]),
    )
    conn.commit()
    cursor.close()
    conn.close()


def add_feedback(data):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "INSERT INTO feedback (id_prediction, label, path, batch, created_at) VALUES (%s, %s, %s, %s, NOW())",
        (data["id"], data["new_identification"], data["image_path"], data["batch"]),
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_count_feedback():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT COUNT(*) AS total_data, MAX(created_at) AS recent_date FROM feedback GROUP BY batch"
    )
    feedback = cursor.fetchall()
    cursor.close()
    conn.close()
    return feedback


def get_feedback():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM feedback group by batch asc")
    feedback = cursor.fetchall()
    cursor.close()
    conn.close()
    return feedback


def get_batch_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM batch order by created_at desc")
    feedback = cursor.fetchall()
    cursor.close()
    conn.close()
    return feedback


def get_feedback_by_batch_id(batch_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM feedback WHERE batch=%s order by created_at desc",
        (batch_id,),
    )
    feedback = cursor.fetchall()
    cursor.close()
    conn.close()
    return feedback


def get_unique_labels_from_table(table_name):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT DISTINCT label FROM %s" % table_name)
    labels = cursor.fetchall()
    cursor.close()
    conn.close()
    return labels
