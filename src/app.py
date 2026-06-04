import joblib as jl
import pandas as pd
import streamlit as st
from shap import TreeExplainer, summary_plot
import matplotlib.pyplot as plt
model = jl.load('best_model.pkl')
X_test_prepped = jl.load('X_test_prepped.pkl')
feature_names = jl.load('feature_names.pkl')
extracted_clf = model.named_steps['model']

months = {
    'January': 1,
   'February': 2,
    'March': 3, 
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September':9,
    'October': 10,
    'November': 11,
    'December': 12}
days_of_week = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3, 
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
  }

if 'history' not in st.session_state:
   st.session_state['history'] = []

st.title('Amazon Shipment Delay Prediction')
with st.expander("See info"):
   st.write("The model has been trained to predict whether a delivery will be fast or not.\n" 
  "\nYou can easily change the values for the delivery details under this message")




explainer = TreeExplainer(extracted_clf)
shap_values = explainer.shap_values(X_test_prepped)
summary_plot(shap_values, X_test_prepped, feature_names=feature_names)
with st.expander('See SHAP'):
   st.pyplot(plt.gcf())
   


with st.form('my_form'):
  col1, col2 = st.columns(2)
  with col1:
    agent_age = st.number_input("Agent's Age", 1, 100, 29)
  with col2:
    agent_rating = st.number_input('Agent Rating', 1.0, 6.0, 4.6, step=0.1)  

  col3, col4 = st.columns(2)
  with col3:
     Weather = st.selectbox('Weather', ['Sunny', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog', 'Windy'])
       
  with col4:
     Traffic = st.selectbox('Traffic', ['High ', 'Jam ', 'Low ', 'Medium '])

  col5, col6 = st.columns(2)
  with col5:
     Vehicle = st.selectbox('Vehicle', ['motorcycle ', 'scooter ', 'van', 'bicycle '])   
  with col6:
      Area = st.selectbox('Area', ['Urban ', 'Metropolitian ', 'Semi-Urban ', 'Other'])
     
  Category = st.selectbox('Category', ['Clothing',  'Electronics',       'Sports',    'Cosmetics',
         'Toys',       'Snacks',        'Shoes',      'Apparel',
      'Jewelry',     'Outdoors',      'Grocery',        'Books',
      'Kitchen',         'Home', 'Pet Supplies',     'Skincare'])
 

  day_of_week = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']) 
  day_of_week = days_of_week[day_of_week]
     
  
  is_weekend = 1 if day_of_week in [6, 7] else 0
  
  col9, col10 = st.columns(2)
  with col9:
   month	= st.selectbox('Month', ['January', 'February', 'March', 'April',
   'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
   month = months[month]

  with col10:
    order_hour_block = st.selectbox('Order Hour Block', ['morning', 'evening', 'afternoon'])

  submitted = st.form_submit_button('Submit')
if submitted:
    to_predict = pd.DataFrame([{
    'Agent_Age': agent_age,
    'Agent_Rating': agent_rating, 
    'Store_Latitude': 17.21,
    'Store_Longitude': 70.66 ,
    'Drop_Latitude': 17.45,
    'Drop_Longitude': 70.82,
    'Weather': Weather,
    'Traffic': Traffic,
    'Vehicle': Vehicle,
    'Area': Area,
    'Category': Category,
    'day_of_week': day_of_week, 
    'is_weekend': is_weekend,
    'month': month,
    'order_hour_block': order_hour_block
    }])
    pred = model.predict(to_predict)
    prob = model.predict_proba(to_predict)
    if pred[0]:
        st.success('The model thinks the delivery will be fast')
    else:
        st.warning('The model does not think the delivery will be fast')
    st.metric('Fast Delivery Probability', f"{round(float(prob[0][1]) * 100, 2)}%")
    # Day 15(
    st.session_state['history'].append({'Agent Rating': agent_rating,
    'Traffic': Traffic,
    'Category': Category,
    'Result': 'Fast' if pred[0] else 'Not Fast',
    'Probability': f"{round(float(prob[0][1]) * 100, 2)}%"})
st.dataframe(st.session_state['history'])
# Day 15)
