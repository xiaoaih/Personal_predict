import streamlit as st
import joblib
import numpy as np
import warnings
import shap
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

warnings.filterwarnings('ignore')

model = joblib.load("D:/Pytorch环境/接单/单四/randomforest.joblib")

with st.form("my_form"):
   slider_shenzang = st.number_input('慢性肾脏病分期(shenzang)')
   slider_age = st.number_input('年龄(age)')
   slider_niaosuan = st.number_input('尿酸(niaosuan)')
   


   submitted = st.form_submit_button("Predict")
   if submitted:
      x_test = np.array([[slider_shenzang,slider_age,slider_niaosuan]])
      explainer = shap.TreeExplainer(model)
      shap_values = explainer.shap_values(x_test)
      shap.force_plot(explainer.expected_value[0], shap_values, x_test,
                      feature_names=['慢性肾脏病分期(shenzang)', '年龄(age)', '尿酸(niaosuan)'], matplotlib=True, show=False)

      plt.tight_layout()
      plt.savefig("outcome_plot.png",dpi=600)
      pred = model.predict(x_test)

      st.markdown("#### _Based on feature values, predicted outcom is {}".format(pred[0]))
      st.image('outcome_plot.png')