import os
import firebase_admin
from firebase_admin import credentials, db

DB_URL = "https://cashreader2026-default-rtdb.asia-southeast1.firebasedatabase.app"

def init_firebase():
    if firebase_admin._apps:
        return

    try:
        cred = credentials.Certificate(
            os.path.join(
                os.path.dirname(__file__),
                "serviceAccountKey.json"
            )
        )

        firebase_admin.initialize_app(
            cred,
            {
                "databaseURL": DB_URL
            }
        )

        print("Firebase connected successfully.")

    except Exception as e:
        print(f"Firebase init failed: {e}")

def get_db(path="/"):
    return db.reference(path)