import streamlit as st
import numpy as np

st.title("Streamlit Demo")

"session state objects:", st.session_state

if 'form_submit' not in st.session_state:
    st.session_state['form_submit'] = False

if 'missing_input' not in st.session_state:
    st.session_state['missing_input'] = False

if 'missing_output' not in st.session_state:
    st.session_state['missing_output'] = False


with st.form("my_form"):
    st.write("Inside the form")
    
    input_path  = st.text_input("input path*", placeholder="Volume(s) directory")
    if st.session_state['missing_input']:
        st.write(":red[INPUT REQUIRED]")

    output_path = st.text_input("putput path*", placeholder="Segmentation(s) directory")
    if st.session_state['missing_output']:
        st.write(":red[OUTPUT REQUIRED]")
    
    slider_val  = st.slider("Form slider")
    checkbox_val = st.checkbox("Form checkbox") 
    
    # Every form must have a submit button. issue https://discuss.streamlit.io/t/forms-submit-button-does-not-work-as-expected/18776
    st.session_state['form_submit'] = st.form_submit_button("Submit")
    if st.session_state['form_submit']:
        st.write("slider", slider_val, "checkbox", checkbox_val)
        st.session_state['missing_input']  = not input_path
        st.session_state['missing_output'] = not output_path
        if st.session_state['missing_input'] or st.session_state['missing_output']:
            st.session_state['form_submit'] = False

if st.session_state['form_submit']:
    # Interactive Streamlit elements, like these sliders, return their value.
    # This gives you an extremely simple interaction model.
    iterations = st.sidebar.slider("Level of detail", 2, 20, 10, 1)
    separation = st.sidebar.slider("Separation", 0.7, 2.0, 0.7885)

    # Non-interactive elements return a placeholder to their location
    # in the app. Here we're storing progress_bar to update it later.
    progress_bar = st.sidebar.progress(0)

    # These two elements will be filled in later, so we create a placeholder
    # for them using st.empty()
    frame_text = st.sidebar.empty()
    image = st.empty()

    test = st.button("test")

    if test:
        st.write("hi")
    else:
        st.write("bye")

    m, n, s = 960, 640, 400
    x = np.linspace(-m / s, m / s, num=m).reshape((1, m))
    y = np.linspace(-n / s, n / s, num=n).reshape((n, 1))

    for frame_num, a in enumerate(np.linspace(0.0, 4 * np.pi, 100)):
        # Here were setting value for these two elements.
        progress_bar.progress(frame_num)
        frame_text.text("Frame %i/100" % (frame_num + 1))

        # Performing some fractal wizardry.
        c = separation * np.exp(1j * a)
        Z = np.tile(x, (n, 1)) + 1j * np.tile(y, (1, m))
        C = np.full((n, m), c)
        M = np.full((n, m), True, dtype=bool)
        N = np.zeros((n, m))

        for i in range(iterations):
            Z[M] = Z[M] * Z[M] + C[M]
            M[np.abs(Z) > 2] = False
            N[M] = i

        # Update the image placeholder by calling the image() function on it.
        image.image(1.0 - (N / N.max()), use_column_width=True)

    # We clear elements by calling empty on them.
    progress_bar.empty()
    frame_text.empty()

    # Streamlit widgets automatically run the script from top to bottom. Since
    # this button is not connected to any other logic, it just causes a plain
    # rerun.
    st.button("Re-run")