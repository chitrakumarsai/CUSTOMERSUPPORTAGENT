import streamlit as st
st.set_page_config(layout='wide', page_title='INVISTA SHOP', page_icon='./image.jpeg')
if 'message_history' not in st.session_state:
    st.session_state.message_history = [{'content':'Hey! I am INVISTA SHOP bot. How can I help you today?', 'type':'assistant'}]

left_col, middle_col, right_col = st.columns([1, 2, 1])

with left_col:
    if st.button('CLEAR CHAT'):
        st.session_state.message_history = []

with middle_col:
    user_input = st.chat_input('Type here....',)
    if user_input:
        st.session_state.message_history.append({'content':user_input, 'type':'user'})

    for i in range(1, len(st.session_state.message_history) + 1):
        this_message = st.session_state.message_history[-i]
        message_box = st.chat_message(this_message['type'])
        message_box.markdown(this_message['content'])

with right_col:
    st.text(st.session_state.message_history)