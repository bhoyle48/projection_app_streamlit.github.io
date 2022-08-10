import streamlit as st

def mysidebar():
    
    ## GET STYLING from Style.css
    with open ("style.css" ) as css:
        st.markdown(f'<style>{css.read()}</style>', unsafe_allow_html=True)
        st.write(f'<style>{css.read()}</style>', unsafe_allow_html=True)
    
    ## SET SIDEBAR DEFINITION
    
    # App About Information
    st.sidebar.title("About")
    st.sidebar.info(
        """
        Web App URL: <https://bhoyle48.github.io/streamlit-projection-app/> \n
        GitHub repository: <https://github.com/bhoyle48/streamlit-projection-app>
        """
    )
    
    # Contact Information
    st.sidebar.title("Contact")
    st.sidebar.info(
        """
        Benjamin Hoyle          
        [GitHub](https://github.com/bhoyle48) | [LinkedIn](https://www.linkedin.com/in/hoyle-benjamin/) | [Email](Mailto:benjamin.hoyle1598@gmail.com?)
        """
    )
