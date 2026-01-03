import time
import main

INTERVAL_SECONDS = 600 

if __name__ == "__main__":
    while True:
        try:
            print("Running pipeline...")
            main.run_pipeline()
            print("Saved data/latest.json")
        except Exception as e:
            print("Pipeline failed:", e)

        time.sleep(INTERVAL_SECONDS)
