import streamlit as st
from routing import route_ticket

st.set_page_config(
    page_title="Complaint Routing Engine",
    layout="wide"
)

st.title("📩 Customer Complaint Classification & Routing Engine")

complaint = st.text_area(
    "Enter Customer Complaint",
    height=200
)

if st.button("Analyze Ticket"):

    result = route_ticket(complaint)

    col1, col2 = st.columns(2)

    with col1:
        st.success("Routing Details")

        st.write("### Queue")
        st.write(result["queue"])

        st.write("### Team")
        st.write(result["team"])

    with col2:
        st.warning("Priority Details")

        st.write("### Priority")
        st.write(result["priority"])

        st.write("### SLA")
        st.write(result["sla"])