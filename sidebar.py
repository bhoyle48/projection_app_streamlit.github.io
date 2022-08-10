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
        Web App URL: [bhoyle48.github.io/forecasting](https://bhoyle48.github.io/forecasting)> \n
        GitHub repository: [github.com/bhoyle48/forecasting](https://github.com/bhoyle48/forecasting)>
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
