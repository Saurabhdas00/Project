import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from math import sqrt
from datetime import datetime

# Load data
customer_df = pd.read_csv("customer_booking_history.csv")
employee_df = pd.read_csv("employee_assignment_data.csv")
cleaning_df = pd.read_csv("hyphomz_indian_cleaning_data.csv")

X_clean = cleaning_df[["bhk", "area_sqft", "has_heavy_furniture", "city_tier", "last_cleaning_days_ago"]]
y_clean = cleaning_df["duration_minutes"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_c, y_train_c)
y_pred_c = knn_model.predict(X_test_c)
knn_rmse = sqrt(mean_squared_error(y_test_c, y_pred_c))
knn_r2 = r2_score(y_test_c, y_pred_c)

X_cust = pd.get_dummies(customer_df[["location_city", "service_type", "preferred_freq"]])
y_cust = customer_df["repeat_customer"]
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_cust, y_cust, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_r, y_train_r)
y_pred_r = rf_model.predict(X_test_r)
rf_accuracy = accuracy_score(y_test_r, y_pred_r)

def get_city_tier(city):
    tier_1 = ['Delhi', 'Mumbai', 'Bengaluru', 'Chennai', 'Kolkata', 'Hyderabad']
    tier_2 = ['Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
    if city in tier_1:
        return 1
    elif city in tier_2:
        return 2
    else:
        return 3

st.set_page_config(page_title="Hyphomz Smart Cleaning Assistant", layout="centered")
st.title("🧼 Hyphomz Smart Cleaning Assistant")

st.header("🔧 Select Customer ID")

with st.form("prediction_form"):
    customer_id_input = st.selectbox("Customer ID", customer_df["customer_id"].unique())
    customer_info = customer_df[customer_df["customer_id"] == customer_id_input].sort_values("booking_date", ascending=False).iloc[0]

    service_type = st.selectbox("Service Type", employee_df["service_type"].unique(), index=list(employee_df["service_type"].unique()).index(customer_info["service_type"]))
    preferred_freq = st.selectbox("Preferred Frequency", ["On-demand", "Weekly", "Bi-weekly"], index=["On-demand", "Weekly", "Bi-weekly"].index(customer_info["preferred_freq"]))
    city = customer_info["location_city"]
    city_tier = get_city_tier(city)
    st.write(f"📍 City: {city} (Tier {city_tier})")

    bhk = st.slider("BHK", 1, 5, 2)
    area = st.number_input("Area (sqft)", 200, 3000, 900)
    furniture = st.checkbox("Heavy Furniture", value=True)

    today = datetime.today()
    last_cleaning_date = pd.to_datetime(customer_info["booking_date"])
    days_ago = (today - last_cleaning_date).days
    st.write(f"🧾 Days Since Last Cleaning: **{days_ago} days**")

    submitted = st.form_submit_button("🔍 Submit")

if submitted:
    st.header("📊 Predictions")

    st.subheader("👷 Auto-Assign Service Provider")
    available_employees = employee_df[
        (employee_df["service_type"] == service_type) &
        (employee_df["available_now"] == 1) &
        (employee_df["can_be_assigned"] == 1)
    ]
    if not available_employees.empty:
        best_employee = available_employees.sort_values(
            by=["avg_rating", "experience_years", "current_distance_km"],
            ascending=[False, False, True]
        ).iloc[0]
        st.success(f"✅ Assigned to: {best_employee['employee_name']} (Rating: {best_employee['avg_rating']})")
    else:
        st.warning("⚠️ No suitable employee available right now.")

    st.subheader("⏱️ Estimated Cleaning Duration")
    input_data = np.array([[bhk, area, int(furniture), city_tier, days_ago]])
    pred_dur = knn_model.predict(input_data)[0]
    st.info(f"🕒 Estimated Duration: **{int(pred_dur)} minutes**")

    st.subheader("💡 Service Recommendation")
    cust_bookings = customer_df[customer_df["customer_id"] == customer_id_input]
    if not cust_bookings.empty:
        top_service = cust_bookings["service_type"].mode()[0]
        freq = cust_bookings["preferred_freq"].mode()[0]
        st.success(f"📌 Recommended: **{top_service}** ({freq}) based on {len(cust_bookings)} past bookings.")
    else:
        st.warning("⚠️ No booking history found.")

st.markdown("---")
st.header("📘 Model Performance Summary")
st.markdown(f"""
- **KNN Cleaning Duration Prediction**
  - RMSE: **{knn_rmse:.2f} minutes**
  - R² Score: **{knn_r2:.2f}**

- **Random Forest Repeat Customer Prediction**
  - Accuracy: **{rf_accuracy:.2f}**
""")
