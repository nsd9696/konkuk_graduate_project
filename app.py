import streamlit as st
st.title('Denoising(Dehazing) Transformer')
st.write('konkuk graduate project')

col1, col2 = st.beta_columns(2)
st.button('Dehaze!')

with col1:
    st.header("Hazy Image")
    st.image("hazy.jpg")

with col2:
    st.header("Clean Image")
    st.image("target.jpg")
    

