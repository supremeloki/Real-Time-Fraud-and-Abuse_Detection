import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timedelta

def generate_synthetic_ride_event(event_time, user_id, driver_id, ride_id,
                                 latitude_start, longitude_start,
                                 latitude_end, longitude_end,
                                 fare_amount, distance_km, duration_min):
    return {
        "event_id": str(uuid.uuid4()),
        "event_timestamp": event_time.isoformat(),
        "event_type": "ride_completed",
        "user_id": user_id,
        "driver_id": driver_id,
        "ride_id": ride_id,
        "start_location_lat": latitude_start,
        "start_location_lon": longitude_start,
        "end_location_lat": latitude_end,
        "end_location_lon": longitude_end,
        "fare_amount": fare_amount,
        "distance_km": distance_km,
        "duration_min": duration_min,
        "payment_method": "cash" if np.random.rand() < 0.6 else "credit_card",
        "promo_code_used": None,
        "device_info": "android_11",
        "ip_address": f"192.168.1.{np.random.randint(1, 255)}"
    }

def generate_promo_abuse_scenario(num_events=50, start_date=datetime.now()):
    events = []
    fake_user_ids = [f"fake_user_{i}" for i in range(num_events // 5)]
    promo_code = "SNAPP_FAKE_PROMO"
    
    for i in range(num_events):
        user = np.random.choice(fake_user_ids)
        driver = f"driver_{np.random.randint(100, 200)}"
        ride = f"ride_{str(uuid.uuid4())[:8]}"
        time = start_date + timedelta(minutes=i*5)
        
        lat_s, lon_s = 35.72 + np.random.normal(0, 0.01), 51.42 + np.random.normal(0, 0.01)
        lat_e, lon_e = 35.73 + np.random.normal(0, 0.01), 51.43 + np.random.normal(0, 0.01)
        
        fare = np.random.uniform(50000, 100000)
        dist = np.random.uniform(2, 5)
        dur = np.random.uniform(10, 20)
        
        event = generate_synthetic_ride_event(time, user, driver, ride, lat_s, lon_s, lat_e, lon_e, fare, dist, dur)
        event["promo_code_used"] = promo_code
        event["event_type"] = "ride_requested" if np.random.rand() < 0.1 else "ride_completed"
        events.append(event)
    return events

def generate_fake_ride_scenario(num_events=30, start_date=datetime.now()):
    events = []
    fake_driver_id = f"fake_driver_{str(uuid.uuid4())[:4]}"
    
    for i in range(num_events):
        user = f"user_{np.random.randint(1, 100)}"
        ride = f"ride_{str(uuid.uuid4())[:8]}"
        time = start_date + timedelta(minutes=i*7)
        
        lat_s, lon_s = 35.75 + np.random.normal(0, 0.005), 51.40 + np.random.normal(0, 0.005)
        lat_e, lon_e = lat_s + np.random.normal(0, 0.001), lon_s + np.random.normal(0, 0.001) # Very short ride
        
        fare = np.random.uniform(20000, 30000)
        dist = np.random.uniform(0.5, 1.5)
        dur = np.random.uniform(3, 7)
        
        event = generate_synthetic_ride_event(time, user, fake_driver_id, ride, lat_s, lon_s, lat_e, lon_e, fare, dist, dur)
        event["payment_method"] = "cash"
        events.append(event)
    return events

if __name__ == "__main__":
    promo_abuse_events = generate_promo_abuse_scenario(num_events=100)
    fake_ride_events = generate_fake_ride_scenario(num_events=60)
    
    all_events = promo_abuse_events + fake_ride_events
    df = pd.DataFrame(all_events)
    df = df.sort_values(by="event_timestamp").reset_index(drop=True)
    
    df.to_csv("synthetic_fraud_events.csv", index=False)
    print("Synthetic fraud events generated and saved to synthetic_fraud_events.csv")